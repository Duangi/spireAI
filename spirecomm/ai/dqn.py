from spirecomm.communication.action import *
from spirecomm.spire.game import Game
from spirecomm.spire.card import Card
import random
import os
import json
import pandas as pd
import tensorflow as tf
from spirecomm.spire.card import CardManager

import numpy as np
from tensorflow import keras
from collections import deque

class GameStateProcessor:
        def __init__(self, card_manager: CardManager):
            self.card_manager = card_manager  # 依赖CardManager处理卡牌嵌入
    
        def get_hand_vector(self, game: Game):
            """处理手牌向量（原DQNAgent的get_hand_vector）"""
            max_hand_size = 10
            hand_vector = [self.card_manager.get_card_embedding_vector(card) for card in game.hand[:max_hand_size]]
            # 填充零向量（使用numpy数组兼容模型输入）
            zero_vectors = [[0.0] * 25 for _ in range(max_hand_size - len(game.hand))]
            return np.array(hand_vector + zero_vectors)  # 输出形状：(10, 25)
    
        def get_fixed_vector(self, game: Game):
            """处理定长数值向量（如hp、floor等）"""
            # 归一化处理（根据实际游戏数值范围调整分母）
            ncurrent_hp = game.current_hp / 200.0
            nmax_hp = game.max_hp / 200.0
            nfloor = game.floor / 55.0
            nact = game.act / 4.0
            ngold = game.gold / 2000.0
            nascension = game.ascension_level / 20.0
            # 角色类型独热编码（假设最多5种角色）
            character_onehot = [0] * 5
            if game.character:
                character_onehot[game.character.value] = 1
            return np.array([ncurrent_hp, nmax_hp, nfloor, nact, ngold, nascension] + character_onehot)
    
        # 可扩展其他预处理方法（如buff、遗物等）
        def get_buff_vector(self, game: Game):
            """示例：处理buff向量（需根据实际buff属性实现）"""
            max_buffs = 5
            buff_vector = []
            for buff in game.buffs[:max_buffs]:
                # 假设每个buff转换为10维向量（如id、层数、持续时间等）
                buff_vector.append([buff.id, buff.stack, buff.duration] + [0.0] * 7)
            # 填充零向量
            while len(buff_vector) < max_buffs:
                buff_vector.append([0.0] * 10)
            return np.array(buff_vector)  # 输出形状：(5, 10)
class DQNAgent:
    def __init__(self, use_dqn=False):  # 新增use_dqn配置项（默认不启用）
        self.game = Game()
        self.card_manager = CardManager()
        self.all_commands_dict = {}
        self.use_dqn = use_dqn  # 新增：是否启用DQN
        # 初始化数据处理器
        self.state_processor = GameStateProcessor(card_manager=self.card_manager)
        # 初始化DQN时传递action_size和state_processor
        self.dqn = DQN(
            action_size=len(self.all_commands_dict),
            state_processor=self.state_processor  # 传递数据处理器
        )

        self.commands_file = os.path.join(os.path.dirname(__file__),"commands.json")
        if os.path.exists(self.commands_file):
            with open(self.commands_file, 'r') as f:
                self.all_commands_dict = json.load(f)
        else:
            with open(self.commands_file, 'w') as f:
                json.dump(self.all_commands_dict, f)


    def generate_action_space(self, game_state):

        self.game = game_state
        commands = game_state.available_commands
        # 将commands数组中的'key','click','state','wait'删掉
        commands = [command for command in commands if command not in ['key', 'click', 'state', 'wait']]
        # 补丁：当在休息站时，选择了升级，但是在选择卡牌时，没有选择卡牌，直接选择了取消，导致收不到新的available_commands，导致无法继续游戏
        # 补丁2：在给精灵献卡时也会出现类似的问题。
        # 如果同时存在cancel和confirm，那么删掉cancel
        if 'cancel' in commands and 'confirm' in commands:
            commands.remove('cancel')
        all_choices = []

        for command in commands:
            command = command.lower()
            index = self.get_command_index(command)
            if command not in ['play', 'potion', 'choose']:
                all_choices.append(Action(command))
            elif command == 'play':
                for i in range(len(game_state.hand)):
                    if game_state.hand[i].is_playable:
                        if game_state.hand[i].has_target:
                            available_monsters = [monster for monster in game_state.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
                            for monster in available_monsters:
                                all_choices.append(Action(command + ' ' + str(i+1) + ' ' + str(monster.monster_index)))
                        else:
                            all_choices.append(Action(command + ' ' + str(i+1)))
            elif command == 'potion':
                potions = game_state.potions
                for i in range(len(potions)):
                    if potions[i].can_discard:
                        all_choices.append(Action(command + ' discard ' + str(i)))
                    if potions[i].can_use:
                        if potions[i].requires_target:
                            for j in range(len(game_state.monsters)):
                                all_choices.append(Action(command + ' use ' + str(i) + ' ' + str(j)))
                        else:
                            all_choices.append(Action(command + ' use ' + str(i)))
            elif command == 'choose':
                if game_state.choice_available:
                    for i in range(len(game_state.choice_list)):
                        all_choices.append(Action(command + ' ' + str(i)))
        return all_choices
    def handle_error(self, error):
        raise Exception(error)
    
    def get_command_index(self, command):
        if command in self.all_commands_dict:
            return self.all_commands_dict[command]
        else:
            return self.add_command(command)
        
    def add_command(self, command):
        if command not in self.all_commands_dict:
            self.all_commands_dict[command] = len(self.all_commands_dict)
            with open(self.commands_file, 'w') as f:
                json.dump(self.all_commands_dict, f)
        
        return self.all_commands_dict[command]
    # 生成输入维度
    

    def get_output_vector(self):
        commands = ["choose", "return", "play", "end", "proceed", "skip", "potion", "leave", "confirm", "cancel"]
        # 使用one-hot编码
        df = pd.DataFrame(commands, columns=['command'])
        one_hot = pd.get_dummies(df['command'])
        return one_hot
    def card_to_vector(self, card:Card):
        return self.card_manager.get_card_embedding_vector(card)

    def random_choose_action(self, all_choices):
        """独立随机选择动作的函数（原随机逻辑）"""
        return random.choice(all_choices)

    def dqn_choose_action(self, game_state, all_choices):
        # 通过state_processor获取预处理后的状态
        fixed_vector = self.state_processor.get_fixed_vector(game_state)
        hand_vector = self.state_processor.get_hand_vector(game_state)
        state = [np.array([fixed_vector]), np.array([hand_vector])]
        action_idx = self.dqn.act(state)
        return all_choices[action_idx] if action_idx < len(all_choices) else random.choice(all_choices)

    def get_next_action_in_game(self, game_state):
        all_choices = self.generate_action_space(game_state)
        if not all_choices:
            return None  # 无可用动作时返回空
        
        # 根据配置选择策略
        if self.use_dqn and len(all_choices) > 0:
            return self.dqn_choose_action(game_state, all_choices)
        else:
            return self.random_choose_action(all_choices)  # 默认使用随机选择
    
    def change_class(self, new_class):
        self.chosen_class = new_class
    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)
    
class DQN:
    def __init__(self, action_size, state_processor: GameStateProcessor):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_max = 0.995
        self.epsilon_decay = 0.995
        self.action_size = action_size
        self.state_processor = state_processor  # 接收数据处理器
        self.model = self._build_model()

    def remember(self, game_state, action, reward, next_game_state, done):
        # 使用state_processor预处理状态
        state = self.state_processor.get_state_vector(game_state)
        next_state = self.state_processor.get_state_vector(next_game_state)
        self.memory.append((state, action, reward, next_state, done))

    def _build_model(self):
        input1 = keras.layers.Input(shape=(7,))
        input2 = keras.layers.Input(shape=(10,25))

        x1 = keras.layers.Dense(64, activation='relu')(input1)
        x1 = keras.layers.Flatten()(x1)

        x2 = keras.layers.Dense(64, activation='relu')(input2)
        x2 = keras.layers.Flatten()(x2)

        combined = keras.layers.concatenate([x1, x2])

        # 修正：使用传入的action_size作为输出维度（替代all_commands_dict）
        output = keras.layers.Dense(self.action_size, activation='softmax')(combined)

        model = keras.models.Model(inputs=[input1, input2], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        # 修正：使用元组存储（保持顺序）
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                # 修正：使用self.gamma替代未定义的discount_factor
                Q_future = max(self.model.predict(next_state, verbose=0)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        """修正：确保输入状态格式正确"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)  # 静默预测
        return np.argmax(act_values[0])