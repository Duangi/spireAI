import torch.nn as nn
import torch.nn.functional as F
from spirecomm.ai.constants import MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_DECK_SIZE, MAX_POTION_COUNT
from spirecomm.ai.dqn_core.action import NUM_ACTION_TYPES

class DQNModel(nn.Module): # 这实际上是一个 Dueling Branching Q-Network
    def __init__(self, state_size):
        super(DQNModel, self).__init__()
        # 共享主体 (Shared Body)
        self.shared_layer1 = nn.Linear(state_size, 512)
        self.shared_layer2 = nn.Linear(512, 512)
        self.shared_layer3 = nn.Linear(512, 256)

        # --- 优势函数 (Advantage) 的“头” ---
        # 评估每个动作/参数相对于平均水平的“优势”
        self.advantage_action_type = nn.Linear(256, NUM_ACTION_TYPES)
        self.advantage_play_card = nn.Linear(256, MAX_HAND_SIZE)
        self.advantage_target_monster = nn.Linear(256, MAX_MONSTER_COUNT)
        self.advantage_choose_option = nn.Linear(256, MAX_DECK_SIZE)
        self.advantage_potion = nn.Linear(256, MAX_POTION_COUNT)

        # --- 状态价值函数 (Value) 的“头” ---
        # 只评估当前状态的好坏，与具体动作无关
        self.value_head = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.shared_layer1(state))
        x = F.relu(self.shared_layer2(x))
        x = F.relu(self.shared_layer3(x))

        # 计算状态价值 V(s)
        value = self.value_head(x)

        # 计算每个动作/参数的优势 A(s, a)
        adv_action_type = self.advantage_action_type(x)
        adv_play_card = self.advantage_play_card(x)
        adv_target_monster = self.advantage_target_monster(x)
        adv_choose_option = self.advantage_choose_option(x)
        adv_potion = self.advantage_potion(x)

        # Dueling DQN 核心公式: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # 组合得到最终的Q值
        # 主动作头的Q值
        action_type_q = value + (adv_action_type - adv_action_type.mean(dim=1, keepdim=True))
        # 参数头的Q值，通常直接用优势值或加上状态价值
        play_card_q = adv_play_card
        target_monster_q = adv_target_monster
        choose_option_q = adv_choose_option
        potion_q = adv_potion

        # 输出类型:action_type_q: [batch_size, NUM_ACTION_TYPES]
        # 其他参数头: [batch_size, param_size]
        return action_type_q, {
            'play_card': play_card_q, 'target_monster': target_monster_q, 
            'choose_option': choose_option_q, 'potion': potion_q
        }