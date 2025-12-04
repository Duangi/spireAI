from calendar import c
import random
from collections import deque
from re import purge
import numpy as np
from sympy import false
import torch
import torch.optim as optim
import torch.nn as nn

from spirecomm.ai.dqn_core.action import BaseAction, DecomposedActionType, PlayAction, ChooseAction, PotionDiscardAction, PotionUseAction, SingleAction, ActionType
from spirecomm.ai.dqn_core.model import DQNModel
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.spire.card import Card
from spirecomm.spire.game import Game
from spirecomm.spire.potion import Potion
from spirecomm.spire.relic import Relic
from spirecomm.spire.screen import ScreenType, ShopScreen

class DQN:
    def __init__(self, state_size, state_processor):
        self.state_size = state_size
        self.memory = deque(maxlen=2000) # 经验回放池，使用固定大小的双端队列
        self.gamma = 0.95    # 折扣因子
        # --- Boltzmann 探索参数 ---
        self.temperature = 5.0  # 初始温度，高温度意味着更随机的探索
        self.temperature_min = 0.1 # 最低温度
        self.temperature_decay = 0.999 # 温度衰减率
        # -------------------------
        self.is_training = True # 默认为训练模式
        self.state_processor = state_processor
        
        self.visited_shop = False  # 用于跟踪是否已经访问过商店

        # --- 神经网络 ---
        # 策略网络 (Policy Network): 用于决定下一步动作，我们会频繁更新它
        self.policy_net = DQNModel(state_size)
        # 目标网络 (Target Network): 用于计算目标Q值，它的权重是定期从策略网络复制过来的，用于稳定训练
        self.target_net = DQNModel(state_size)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # 初始化时，权重完全相同
        self.target_net.eval() # 目标网络只用于推理，不进行训练
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.loss_fn = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        state_numpy = state.detach().cpu().numpy()
        next_state_numpy = next_state.detach().cpu().numpy()
        self.memory.append((state_numpy, action, reward, next_state_numpy, done))

    def train(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        
        # 将经验元组解压并转换为PyTorch张量
        states = torch.from_numpy(np.vstack([e[0] for e in minibatch if e is not None])).float()
        actions = [e[1] for e in minibatch if e is not None]
        rewards = torch.tensor([e[2] for e in minibatch if e is not None]).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch if e is not None])).float()
        dones = torch.tensor([e[4] for e in minibatch if e is not None]).float()

        # --- 1. 计算预测Q值 (Predicted Q-values) ---
        # 使用策略网络(policy_net)获取当前状态的Q值
        pred_action_q, pred_arg_q = self.policy_net(states)
        
        # 根据实际执行的动作，从多头输出中提取对应的Q值
        predicted_q_values = []
        for i, action in enumerate(actions):
            # 动作类型的Q值
            decomposed_type = action.decomposed_type
            # 使用安全索引，避免 action.decomposed_type.value 为字符串导致 TypeError
            action_idx = self._action_index_from_decomposed(action)
            q_val = pred_action_q[i, action_idx]
            # 如果动作有参数，加上参数的Q值
            if isinstance(action, PlayAction):
                q_val += pred_arg_q['play_card'][i, action.hand_idx]
                if action.target_idx is not None:
                    q_val += pred_arg_q['target_monster'][i, action.target_idx]
            elif isinstance(action, ChooseAction):
                q_val += pred_arg_q['choose_option'][i, action.choice_idx]
            elif isinstance(action, PotionUseAction):
                q_val += pred_arg_q['potion'][i, action.potion_idx]
                if action.target_idx is not None:
                    q_val += pred_arg_q['target_monster'][i, action.target_idx]
            predicted_q_values.append(q_val)
        predicted_q_values = torch.stack(predicted_q_values)

        # --- 2. 计算目标Q值 (Target Q-values) ---
        # 使用目标网络(target_net)来计算下一状态的最大Q值，这可以稳定训练过程
        with torch.no_grad():
            next_action_q, next_arg_q = self.target_net(next_states)
            # 找到下一状态中，Q值最高的合法动作类型
            # 注意：这里为了简化，我们只考虑了下一状态的动作类型Q值，
            # TODO 可能更精确的方法更好，但我觉得估计这样应该就行了
            # 一个更精确的方法是找到下一状态Q值最高的完整动作（类型+参数）
            max_next_q = next_action_q.max(1)[0]
            
        # 贝尔曼方程: Target = reward + gamma * max_next_q
        # 如果是回合结束(done=True)，则没有未来奖励，Target = reward
        target_q_values = rewards + (1 - dones) * self.gamma * max_next_q

        # --- 3. 计算损失并进行反向传播 ---
        loss = self.loss_fn(predicted_q_values, target_q_values)
        
        # 梯度清零
        self.optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新策略网络的权重
        self.optimizer.step()

        # 在每次训练后衰减温度
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

    def set_inference_mode(self):
        """切换到推理模式，不进行探索。"""
        self.is_training = False

    def choose_action(self, state_tensor, masks, game_state:Game):
        """
        使用 Boltzmann 探索 (Softmax 探索) 来选择一个分解式动作。
        Q值越高的动作被选择的概率越大。
        :param state_tensor: 当前状态的 PyTorch 张量。
        :param masks: 一个包含所有合法动作掩码的字典。
        :return: 一个结构化的动作对象 (e.g., PlayAction, ChooseAction)。
        """
        action_type_q, arg_q = self.get_q_values(state_tensor, use_policy_net=True)

        # 1. 决策第一步：选择动作类型 (Action Type)
        action_type_q = action_type_q.squeeze(0) # 移除 batch 维度
        action_type_mask = torch.from_numpy(masks['action_type']).bool()
        
        # 应用掩码，将非法动作的Q值设为负无穷
        action_type_q[~action_type_mask] = -float('inf')
        
        # 如果所有动作类型都被屏蔽了
        if not action_type_mask.any():
            raise ValueError("所有动作类型均被屏蔽，无法选择动作。")

        if self.is_training:
            # 训练模式：Boltzmann 探索
            action_type_probs = torch.softmax(action_type_q / self.temperature, dim=-1)
            # 从概率分布中采样一个动作类型
            action_type_idx = torch.multinomial(action_type_probs, 1).item()
        else:
            # 推理模式：选择Q值最高的动作
            action_type_idx = torch.argmax(action_type_q).item()

        action_type = DecomposedActionType(action_type_idx)

        # 2. 决策第二步：根据动作类型选择参数
        if action_type == DecomposedActionType.PLAY:
            # 需要选择打哪张牌 (play_card) 和目标 (target_monster)
            play_card_q = arg_q['play_card'].squeeze(0)
            play_card_mask = torch.from_numpy(masks['play_card']).bool()
            play_card_q[~play_card_mask] = -float('inf')
            if self.is_training:
                play_card_probs = torch.softmax(play_card_q / self.temperature, dim=-1)
                card_idx = torch.multinomial(play_card_probs, 1).item()
            else:
                card_idx = torch.argmax(play_card_q).item()

            # 选中了牌之后，判断这个牌是否需要目标
            target_idx = None
            chosen_card:Card = game_state.hand[card_idx]
            if chosen_card.has_target:
                target_q = arg_q['target_monster'].squeeze(0)
                target_mask = torch.from_numpy(masks['target_monster']).bool()
                target_q[~target_mask] = -float('inf')
                if self.is_training:
                    target_probs = torch.softmax(target_q / self.temperature, dim=-1)
                    target_idx = torch.multinomial(target_probs, 1).item()
                else:
                    target_idx = torch.argmax(target_q).item()

            # 如果目标掩码全为False，说明该牌无需目标
            if not np.any(masks['target_monster']):
                target_idx = None

            return PlayAction(type=ActionType.PLAY, hand_idx=card_idx, target_idx=target_idx)

        elif action_type == DecomposedActionType.CHOOSE:
            # 当遇到商店选项的时候，首先要看需不需要进入商店
            if game_state.screen_type == ScreenType.SHOP_ROOM:
                if not self.visited_shop:
                    self.visited_shop = True
                    for i, option in enumerate(game_state.choice_list):
                        if option == 'shop':
                            return ChooseAction(type=ActionType.CHOOSE, choice_idx=i)
                else:
                    self.visited_shop = False
                    return SingleAction(type=ActionType.PROCEED, decomposed_type=ActionType.PROCEED)
            # 实际上是不需要看买不买得起的，这里纯粹是因为没有leave的选项导致走不开商店导致的。
            # 进入商店，弹出商店的Screen后，继续选择选项
            # if game_state.screen_type == ScreenType.SHOP_SCREEN:
            #     # 这里需要认定screen_type是SHOP_SCREEN才能继续选择商店选项
            #     shop_screen:ShopScreen = game_state.screen
            #     # 把所有物品中的最便宜的价格找出来
            #     purge_cost = shop_screen.purge_cost
            #     if purge_cost > game_state.gold:
            #         # 买不起移除牌，把mask对应的位置设为-inf
            #         index = self.choose_index_based_name(game_state.choice_list, 'purge')
            #         masks['choose_option'][index] = false
            #     for i, card in enumerate(shop_screen.cards):
            #         card:Card = shop_screen.cards[i]
            #         if card.cost > game_state.gold:
            #             # 找到买不起的牌，把mask对应的位置设为-inf
            #             index = self.choose_index_based_name(game_state.choice_list, card.name)
            #             masks['choose_option'][index] = false
            #     for i, relic in enumerate(shop_screen.relics):
            #         relic:Relic = shop_screen.relics[i]
            #         if relic.price > game_state.gold:
            #             # 找到买不起的遗物，把mask对应的位置设为-inf
            #             index = self.choose_index_based_name(game_state.choice_list, relic.name)
            #             masks['choose_option'][index] = false
            #     for i, potion in enumerate(shop_screen.potions):
            #         potion:Potion = shop_screen.potions[i]
            #         if potion.price > game_state.gold:
            #             # 找到买不起的药水，把mask对应的位置设为-inf
            #             index = self.choose_index_based_name(game_state.choice_list, potion.name)
            #             masks['choose_option'][index] = false
            #     # 如果choose_masks全部是false，说明买不起任何东西，只能选择离开
            #     if not np.any(masks['choose_option']):
            #         return SingleAction(type=ActionType.CANCEL, decomposed_type=ActionType.CANCEL)
                

            choose_q = arg_q['choose_option'].squeeze(0)
            choose_mask = torch.from_numpy(masks['choose_option']).bool()
            choose_q[~choose_mask] = -float('inf')
            if self.is_training:
                choose_probs = torch.softmax(choose_q / self.temperature, dim=-1)
                choice_idx = torch.multinomial(choose_probs, 1).item()
            else:
                choice_idx = torch.argmax(choose_q).item()
            return ChooseAction(type=ActionType.CHOOSE, choice_idx=choice_idx)

        elif action_type == DecomposedActionType.POTION_USE:
            # 需要选择使用哪个药水，以及可能的目标
            potion_q = arg_q['potion'].squeeze(0)
            potion_mask = torch.from_numpy(masks['potion']).bool()
            potion_q[~potion_mask] = -float('inf')
            if self.is_training:
                potion_probs = torch.softmax(potion_q / self.temperature, dim=-1)
                potion_idx = torch.multinomial(potion_probs, 1).item()
            else:
                potion_idx = torch.argmax(potion_q).item()

            target_idx = None
            # 检查使用该药水是否需要目标
            chosen_potion:Potion = game_state.potions[potion_idx]
            if chosen_potion.requires_target:
                target_q = arg_q['target_monster'].squeeze(0)
                target_mask = torch.from_numpy(masks['target_monster']).bool()
                target_q[~target_mask] = -float('inf')
                if self.is_training:
                    target_probs = torch.softmax(target_q / self.temperature, dim=-1)
                    target_idx = torch.multinomial(target_probs, 1).item()
                else:
                    target_idx = torch.argmax(target_q).item()
            
            return PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=target_idx)
        
        elif action_type == DecomposedActionType.POTION_DISCARD:
            # 需要选择丢弃哪个药水
            potion_q = arg_q['potion'].squeeze(0)
            # 假设丢弃药水的合法性掩码与使用药水相同
            potion_mask = torch.from_numpy(masks['potion']).bool()
            potion_q[~potion_mask] = -float('inf')
            if self.is_training:
                potion_probs = torch.softmax(potion_q / self.temperature, dim=-1)
                potion_idx = torch.multinomial(potion_probs, 1).item()
            else:
                potion_idx = torch.argmax(potion_q).item()
            return PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx)
            
        # 对于所有无参数的动作 (END, PROCEED, CANCEL, LEAVE, CONFIRM)
        else:
            # 这些动作没有参数，直接构建并返回SingleAction对象
            # 我们需要一个方法将DecomposedActionType映射回ActionType
            base_action_type = action_type.to_action_type()
            if base_action_type is None:
                raise ValueError(f"无法将有参数的分解动作 {action_type.name} 转换为 SingleAction")
            return SingleAction(type=base_action_type, decomposed_type=action_type)

    def choose_index_based_name(self, choice_list, name):
        """根据名称选择对应的索引"""
        for i, choice in enumerate(choice_list):
            if choice == name:
                return i
        return None
    def get_q_values(self, state, use_policy_net=True):
        """获取所有头的Q值"""
        with torch.no_grad():
            if use_policy_net:
                self.policy_net.eval()
                action_q, arg_q = self.policy_net(state)
                if self.is_training: # 只有在训练模式下才切换回 .train()
                    self.policy_net.train()
            else: # 使用目标网络
                action_q, arg_q = self.target_net(state)
        return action_q, arg_q

    def get_all_legal_action_q_values(self, state_tensor, game_state):
        """
        获取当前状态下所有合法动作的Q值，用于日志记录。
        :param state_tensor: 当前状态的 PyTorch 张量。
        :param game_state: 原始游戏状态字典，用于获取合法动作。
        :return: 一个字典，键是动作字符串，值是对应的Q值。
        """
        q_values = {}
        available_commands = self.state_processor.get_available_actions(game_state)
        if not available_commands:
            return q_values

        # 使用策略网络获取Q值
        action_type_q, arg_q = self.get_q_values(state_tensor, use_policy_net=True)
        action_type_q = action_type_q.squeeze(0)
        arg_q = {k: v.squeeze(0) for k, v in arg_q.items()}

        for action in available_commands:
            # 分解式动作没有直接的Q值，我们需要根据其构成来计算
            # 注意：这里的计算方式和训练时的Q值计算方式保持一致
            q_val = 0
            # 动作类型的Q值
            if hasattr(action, 'decomposed_type'):
                q_val += action_type_q[action.decomposed_type.value].item()
            
            # 参数的Q值
            if isinstance(action, PlayAction):
                q_val += arg_q['play_card'][action.hand_idx].item()
                if action.target_idx is not None:
                    q_val += arg_q['target_monster'][action.target_idx].item()
            elif isinstance(action, ChooseAction):
                q_val += arg_q['choose_option'][action.choice_idx].item()
            elif isinstance(action, PotionUseAction):
                q_val += arg_q['potion'][action.potion_idx].item()
                if action.target_idx is not None:
                    q_val += arg_q['target_monster'][action.target_idx].item()
            q_values[action.to_string()] = q_val
        return q_values

    def update_target_net(self):
        """定期将策略网络的权重复制到目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # 新增：把 action.decomposed_type 安全地转换为整数索引
    def _action_index_from_decomposed(self, action):
        """
        返回可用于索引 pred_action_q 的整数索引。
        兼容场景：
          - action.decomposed_type.value 是 int
          - action.decomposed_type.value 是可以转为 int 的字符串
          - action.decomposed_type 是 Enum/IntEnum（通过成员的 value 或顺序）
          - 如果实例上定义了 ACTION_INDEX_MAP / action_index_map 字典，则使用该映射（支持 key 为 name 或 value）
        """
        # 尝试取 value
        dt = getattr(action, "decomposed_type", None)
        val = getattr(dt, "value", None)

        # 直接是整数
        if isinstance(val, int):
            return int(val)
        # value 为可转 int 的字符串
        if isinstance(val, str):
            try:
                return int(val)
            except Exception:
                pass
        # 如果 decomposed_type 本身是 int 或可转为 int
        if isinstance(dt, int):
            return int(dt)
        if isinstance(dt, str):
            try:
                return int(dt)
            except Exception:
                pass