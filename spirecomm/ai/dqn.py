import torch
import json
import sys
from typing import Optional
from spirecomm.ai import absolute_logger
from spirecomm.ai.dqn_core.algorithm import DQN
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.ai.dqn_core.reward import RewardCalculator
from spirecomm.spire import game
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.progress_logger import ProgressLogger
from spirecomm.ai.absolute_logger import AbsoluteLogger
from spirecomm.spire.game import Game
from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases


class DQNAgent:
    """
    一个集决策、记忆、学习于一体的DQN智能体。
    这个类是与spirecomm协调器直接交互的接口。
    """

    def __init__(self, play_mode=False, model_path=None):
        self.play_mode = play_mode
        # 1. 初始化核心组件
        self.state_processor = GameStateProcessor()
        self.reward_calculator = RewardCalculator()
        self.progress_logger = ProgressLogger()
        self.absolute_logger = AbsoluteLogger()
        self.absolute_logger.start_episode()

        # 假设状态向量大小为 10358
        state_size = 10358 
        self.dqn_algorithm = DQN(state_size, self.state_processor)

        if self.play_mode:
            print("--- Agent is in PLAY MODE ---", file=sys.stderr)
            if model_path:
                print(f"Loading model from: {model_path}", file=sys.stderr)
                self.dqn_algorithm.policy_net.load_state_dict(torch.load(model_path))
            self.dqn_algorithm.set_inference_mode()
            self.dqn_algorithm.policy_net.eval() # 游玩模式下，使用评估模式
        else:
            print("--- Agent is in TRAINING MODE ---", file=sys.stderr)
            # 确保在训练模式下，网络处于 .train() 状态
            # 这将允许梯度计算和权重更新
            self.dqn_algorithm.policy_net.train()

        # 2. 用于存储上一步信息的变量
        self.previous_game_state = None
        self.previous_action = None
        self.previous_state_tensor = None

    def get_next_action_in_game(self, game_state:Game):
        """
        这是由Coordinator在游戏状态改变时调用的核心回调函数。
        """
        # --- 学习与记忆 ---
        reward = 0
        # 只有在非游玩模式下，才进行学习
        if not self.play_mode:
            if self.previous_game_state is not None and self.previous_action is not None:
                # a. 计算奖励
                reward = self.reward_calculator.calculate(self.previous_game_state, game_state, self.previous_action)
                
                # b. 处理新状态
                next_state_tensor = self.state_processor.process(game_state)
                # game_state 是 Game 对象，直接访问属性
                done = not game_state.in_game
                
                # c. 记忆经验
                self.dqn_algorithm.remember(self.previous_state_tensor, self.previous_action, reward, next_state_tensor, done)
                
                # d. 训练模型
                self.dqn_algorithm.train()

        # --- 决策 ---
        # 1. 获取当前状态的向量和合法的动作掩码
        current_state_tensor = self.state_processor.process(game_state)
        # game_state 是 Game 对象，直接访问属性
        available_commands = game_state.available_commands
        masks = self.state_processor.get_action_masks(game_state)

        # 2. 如果没有可选动作，直接返回
        if not available_commands:
            # 游戏结束或出现意外情况，结束日志记录
            if self.progress_logger.file_handle:
                self.progress_logger.end_episode()
            return None

        # 3. 使用DQN算法选择一个动作
        chosen_action = self.dqn_algorithm.choose_action(current_state_tensor, masks)

        # 4. 如果算法没有选择任何动作（例如，在状态转换期间），则不执行任何操作
        if chosen_action is None:
            return None

        # --- 可视化日志记录 ---
        if self.previous_game_state is not None:
            # 获取所有合法动作的Q值用于记录
            q_values_log = self.dqn_algorithm.get_all_legal_action_q_values(current_state_tensor, game_state)

            log_info = {
                'q_values': q_values_log,
                'action_taken_at_prev_state': self.previous_action.to_string() if self.previous_action else "None",
                'reward_for_prev_action': reward,
                'chosen_action_for_current_state': chosen_action.to_string(),
                'prev_player': self.previous_game_state.player.to_json() if self.previous_game_state.player else {},
                'next_player': game_state.player.to_json() if game_state.player else {},
                'prev_monsters': [m.to_json() for m in self.previous_game_state.monsters],
                'next_monsters': [m.to_json() for m in game_state.monsters],
                'reward': reward
            }
            self.progress_logger.log_step(log_info)
            self.absolute_logger.write(log_info)
        # --- 为下一步做准备 ---
        # 存储当前的状态和动作用于下一次学习
        self.previous_game_state = game_state
        self.previous_action = chosen_action
        self.previous_state_tensor = current_state_tensor

        # --- 返回动作给协调器 ---
        # 核心改动：协调器需要的是一个可执行的字符串命令，而不是我们的内部动作对象。
        # 我们调用 to_string() 方法将其转换。
        action_string = chosen_action.to_string()
        
        return action_string

    def get_next_action_out_of_game(self, game_state):
        # 在游戏外，我们总是选择开始游戏
        # 如果上一个日志文件还开着，说明游戏异常结束，关闭它
        if self.progress_logger.file_handle:
            self.progress_logger.end_episode()
        self.previous_game_state = None
        self.previous_action = None
        self.previous_state_tensor = None
        self.progress_logger.start_episode() # 新的一局游戏开始
        self.absolute_logger.start_episode()
        return self.state_processor.get_start_game_action(game_state)

    def handle_error(self, error):
        """
        处理来自协调器的错误回调。
        当一个动作无效时，这通常意味着AI对游戏状态的理解出现了偏差。
        一个稳健的策略是清空动作队列，并根据当前最新的游戏状态重新决策。
        """
        print(f"Received error: {error}", file=sys.stderr)
        # 返回 None 会导致协调器清空动作队列，然后在下一个循环中根据最新状态重新调用 get_next_action_in_game
        return None

    def change_class(self, chosen_class: PlayerClass):
        # 这个方法可以被主循环调用，但目前我们的Agent是通用的，所以不需要做什么
        print(f"Changing class to {chosen_class.name}", file=sys.stderr)

    def learn(self, batch_size=32):
        """
        从经验回放区采样数据，训练一次网络。
        这是提供给 train.py 在每局结束后调用的接口。
        """
        self.dqn_algorithm.train(batch_size)