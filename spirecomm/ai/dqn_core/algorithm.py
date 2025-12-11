from calendar import c
from dataclasses import fields
import random
from collections import deque
from re import purge
import numpy as np
from sympy import false
import torch
import torch.optim as optim
import torch.nn as nn
from spirecomm.ai.constants import MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_DECK_SIZE, MAX_POTION_COUNT
import os
import datetime

from spirecomm.ai.absolute_logger import AbsoluteLogger, LogType
from spirecomm.ai.dqn_core.action import BaseAction, DecomposedActionType, PlayAction, ChooseAction, PotionDiscardAction, PotionUseAction, SingleAction, ActionType
from spirecomm.ai.dqn_core.model import DQNModel, SpireConfig, SpireDQN, SpireOutput, SpireState
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.spire.card import Card
from spirecomm.spire.game import Game
from spirecomm.spire.potion import Potion
from spirecomm.spire.relic import Relic
from spirecomm.spire.screen import ScreenType, ShopScreen
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger

class SpireAgent:
    def __init__(self, config: SpireConfig, device="cuda" if torch.cuda.is_available() else "cpu", wandb_logger: WandbLogger = None):
        self.cfg = config
        self.device = device
        self.wandb_logger = wandb_logger
        self.last_q_values = {}
        self.total_steps = 0
        
        # --- 模型初始化 ---
        self.policy_net = SpireDQN(config).to(device)
        self.target_net = SpireDQN(config).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # --- 优化器与损失 ---
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        # Huber Loss 对异常值（比如突然爆发的伤害数值）更稳健
        self.loss_fn = nn.SmoothL1Loss() 
        
        # --- 经验回放 ---
        # 存储格式: (state: SpireState, action: object, reward: float, next_state: SpireState, done: bool)
        self.memory = deque(maxlen=5000) 
        
        # --- 训练超参数 ---
        self.batch_size = 32
        self.gamma = 0.99
        self.temperature = 2.0
        self.temperature_min = 0.1
        # 调整衰减率以适应长期的训练过程
        # 假设每局游戏平均 64 次训练 (TRAIN_BATCHES_PER_EPISODE)，总共 5000 局
        # 总训练步数 = 5000 * 64 = 320,000
        # 如果使用 0.9995，约 6000 步（不到 100 局）就会降到最低温度，过快。
        # 使用 0.99999，在 300,000 步时约为 0.1 (2.0 * 0.99999^300000 ≈ 0.099)
        self.temperature_decay = 0.99999
        self.is_training = True

    def save_model(self, path):
        torch.save({
            'model': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.cfg
        }, path)

    def load_model(self, path):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['model'])
            self.target_net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    # ==========================================
    # 1. 核心数据处理: Collate (打包)
    # ==========================================
    def collate_states(self, states: list[SpireState]) -> SpireState:
        """
        将 List[SpireState] (长度=Batch) 转换为 一个 SpireState (Tensor维度增加Batch维)
        例如: 32 个 [10, 128] 的 hand_vec -> 1 个 [32, 10, 128] 的 hand_vec
        """
        # 动态获取 SpireState 的所有字段名
        field_names = [f.name for f in fields(SpireState)]
        
        batched_data = {}
        for name in field_names:
            # 提取 list 中每个 state 的对应字段
            tensors = [getattr(s, name) for s in states]
            
            # 堆叠 (Stack)
            # 注意: 所有输入 Tensor 必须在 CPU 上，这一步还没进 GPU
            batched_tensor = torch.stack(tensors, dim=0) 
            batched_data[name] = batched_tensor
            
        # 返回一个新的 SpireState 对象
        return SpireState(**batched_data)

    def to_device(self, state: SpireState) -> SpireState:
        """将 SpireState 中的所有 Tensor 移动到 GPU"""
        new_data = {}
        for name in [f.name for f in fields(SpireState)]:
            val = getattr(state, name)
            if isinstance(val, torch.Tensor):
                new_data[name] = val.to(self.device)
        return SpireState(**new_data)

    # ==========================================
    # 2. 经验存取
    # ==========================================
    def remember(self, state: SpireState, action, reward, next_state: SpireState, done: bool, reward_details: str = ""):
        """
        存入经验池。
        注意：存入前不要 to(device)，保持在 CPU 上以节省显存。
        """
        # 这里的 state 应该是单个对象的 SpireState (非 Batch)
        self.memory.append((state, action, reward, next_state, done))
        self.total_steps += 1
        
        if self.wandb_logger:
            try:
                # Format action string if it's an object
                if hasattr(action, 'to_string'):
                    action_desc = action.to_string()
                elif isinstance(action, str):
                    action_desc = action
                else:
                    action_desc = str(action)

                self.wandb_logger.log_step(
                    step_count=self.total_steps,
                    state=state,
                    action_desc=action_desc,
                    q_values=self.last_q_values,
                    reward=reward,
                    reward_details=reward_details
                )

                if done:
                    self.wandb_logger.commit_table(clear_buffer=True, step=self.total_steps)
            except Exception as e:
                raise RuntimeError(f"WandB Logging Error: {e}")
                pass

    # ==========================================
    # 3. 训练循环
    # ==========================================
    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # 1. 采样
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 解压
        state_list = [x[0] for x in minibatch]
        action_list = [x[1] for x in minibatch]
        reward_list = [x[2] for x in minibatch]
        next_state_list = [x[3] for x in minibatch]
        done_list = [x[4] for x in minibatch]

        # 2. 打包并移动到 GPU
        batch_state = self.to_device(self.collate_states(state_list))
        batch_next_state = self.to_device(self.collate_states(next_state_list))
        
        batch_rewards = torch.tensor(reward_list, device=self.device, dtype=torch.float32)
        batch_dones = torch.tensor(done_list, device=self.device, dtype=torch.float32)

        # 3. 计算当前 Q 值 (Predicted Q)
        # Forward pass
        output: SpireOutput = self.policy_net(batch_state)
        
        # 提取对应动作的 Q 值
        pred_q_values = []
        
        for i, action in enumerate(action_list):
            # 获取 Action Type 的索引 (int)
            # 假设 action 有 decomposed_type 属性
            act_idx = action.decomposed_type.value 
            
            # 基础 Q 值 (Action Type Q)
            q_val = output.q_action_type[i, act_idx]
            
            # 加上分支 Q 值 (Argument Q)
            if isinstance(action, PlayAction):
                # play_card: [Batch, 10] -> 取第 i 个样本的第 hand_idx 张牌
                # 确保索引在合法范围内
                hand_idx = min(action.hand_idx, 9) # MAX_HAND_SIZE-1
                q_val += output.q_play_card[i, hand_idx]
                if action.target_idx is not None:
                    target_idx = min(action.target_idx, 4) # MAX_MONSTER_COUNT-1
                    q_val += output.q_target_monster[i, target_idx]
                    
            elif isinstance(action, ChooseAction):
                # 确保索引在合法范围内
                choice_idx = min(action.choice_idx, 14) # MAX_CHOICE_LIST-1
                q_val += output.q_choose_option[i, choice_idx]
                
            elif isinstance(action, PotionUseAction):
                pot_idx = min(action.potion_idx, 4) # MAX_POTION_COUNT-1
                q_val += output.q_potion_use[i, pot_idx]
                if action.target_idx is not None:
                    target_idx = min(action.target_idx, 4)
                    q_val += output.q_target_monster[i, target_idx]
            
            elif isinstance(action, PotionDiscardAction):
                pot_idx = min(action.potion_idx, 4)
                q_val += output.q_potion_discard[i, pot_idx]
                
            pred_q_values.append(q_val)
            
        # 堆叠成 Tensor [Batch]
        pred_q_tensor = torch.stack(pred_q_values)

        # 4. 计算目标 Q 值 (Target Q)
        with torch.no_grad():
            next_output: SpireOutput = self.target_net(batch_next_state)
            
            # 简化策略：Double DQN 或直接 Max
            # 这里取 Next Action Type 的最大值作为估计
            # 改进方向：应该取 Max(ActionType + Max(Branch))，但这计算比较复杂
            # 目前 heuristic: Max Action Type Q 已经能提供足够的梯度方向
            max_next_q, _ = next_output.q_action_type.max(dim=1)
            
            target_q_tensor = batch_rewards + (1 - batch_dones) * self.gamma * max_next_q

        # 5. 反向传播
        loss = self.loss_fn(pred_q_tensor, target_q_tensor)
        
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪防止梯度爆炸 (Transformer/LSTM常用)
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.wandb_logger:
            # 过滤掉可能存在的 -inf / inf，避免污染 Log 统计
            valid_q = pred_q_tensor[torch.isfinite(pred_q_tensor)]
            if valid_q.numel() > 0:
                avg_q = valid_q.mean().item()
                max_q = valid_q.max().item()
            else:
                avg_q = 0.0
                max_q = 0.0

            self.wandb_logger.log_metrics(
                {
                    "loss": loss.item(),
                    "avg_reward": batch_rewards.mean().item(),
                    "avg_q_value": avg_q,
                    "max_q_value": max_q,
                    "temperature": self.temperature
                },
                step=self.total_steps
            )

        # 6. 更新温度
        if self.temperature > self.temperature_min:
            self.temperature *= self.temperature_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ==========================================
    # 4. 动作选择 (Inference)
    # ==========================================
    def choose_action(self, state: SpireState, masks, game_state_obj: Game) -> BaseAction:
        """
        state: 单个 SpireState (CPU)
        masks: 动作掩码字典
        game_state_obj: 原始 Game 对象 (用于逻辑校验/掩码处理)
        """
        # 1. 增加 Batch 维度并移至 GPU
        # collate_states([state]) -> [1, ...]
        batch_state = self.to_device(self.collate_states([state]))
        
        with torch.no_grad():
            output: SpireOutput = self.policy_net(batch_state)
        
        # 提取 Batch 0 的结果
        q_type = output.q_action_type[0]      # [Num_Types]
        q_card = output.q_play_card[0]        # [10]
        q_monster = output.q_target_monster[0]# [5]
        q_choice = output.q_choose_option[0]  # [15]
        q_pot_use = output.q_potion_use[0]    # [5]
        q_pot_disc = output.q_potion_discard[0] # [5]

        # --- Wandb Logging: Store Q-values ---
        if self.wandb_logger:
            try:
                self.last_q_values = {
                    'action_type': {},
                    'play_card': {},
                    'target_monster': {},
                    'choose_option': {},
                    'potion_use': {},
                    'potion_discard': {}
                }
                
                # 1. Action Types
                at_q = q_type.detach().cpu().numpy()
                at_mask = masks.get('action_type', np.ones_like(at_q, dtype=bool))
                for i, val in enumerate(at_q):
                    if i < len(at_mask) and at_mask[i]:
                        try:
                            name = DecomposedActionType(i).name
                        except:
                            name = f"Type_{i}"
                        self.last_q_values['action_type'][name] = float(val)

                # 2. Arguments
                # Play Card
                pc_q = q_card.detach().cpu().numpy()
                pc_mask = masks.get('play_card', np.zeros_like(pc_q, dtype=bool))
                for i, val in enumerate(pc_q):
                    if i < len(pc_mask) and pc_mask[i]:
                        card_name = f"Card_{i}"
                        if i < len(game_state_obj.hand):
                            card_name = f"{game_state_obj.hand[i].name}"
                        self.last_q_values['play_card'][card_name] = float(val)

                # Target Monster
                tm_q = q_monster.detach().cpu().numpy()
                tm_mask = masks.get('target_monster', np.zeros_like(tm_q, dtype=bool))
                for i, val in enumerate(tm_q):
                    if i < len(tm_mask) and tm_mask[i]:
                        mon_name = f"Monster_{i}"
                        if i < len(game_state_obj.monsters):
                            mon_name = f"{game_state_obj.monsters[i].name}"
                        self.last_q_values['target_monster'][mon_name] = float(val)

                # Choose Option
                co_q = q_choice.detach().cpu().numpy()
                co_mask = masks.get('choose_option', np.zeros_like(co_q, dtype=bool))
                for i, val in enumerate(co_q):
                    if i < len(co_mask) and co_mask[i]:
                        opt_name = f"Choice_{i}"
                        if hasattr(game_state_obj, 'choice_list') and i < len(game_state_obj.choice_list):
                            opt_name = f"{game_state_obj.choice_list[i]}"
                        self.last_q_values['choose_option'][opt_name] = float(val)

                # Potion Use
                pu_q = q_pot_use.detach().cpu().numpy()
                pu_mask = masks.get('potion_use', np.zeros_like(pu_q, dtype=bool))
                for i, val in enumerate(pu_q):
                    if i < len(pu_mask) and pu_mask[i]:
                        pot_name = f"PotUse_{i}"
                        if i < len(game_state_obj.potions):
                            pot_name = f"{game_state_obj.potions[i].name}"
                        self.last_q_values['potion_use'][pot_name] = float(val)

                # Potion Discard
                pd_q = q_pot_disc.detach().cpu().numpy()
                pd_mask = masks.get('potion_discard', np.zeros_like(pd_q, dtype=bool))
                for i, val in enumerate(pd_q):
                    if i < len(pd_mask) and pd_mask[i]:
                        pot_name = f"PotDisc_{i}"
                        if i < len(game_state_obj.potions):
                            pot_name = f"{game_state_obj.potions[i].name}"
                        self.last_q_values['potion_discard'][pot_name] = float(val)

            except Exception as e:
                self.last_q_values = {}
        # -------------------------------------

        # 判断药水是否满了
        # 如果状态里面的choice_list有potion字段的话，把对应的index选出来，mask置为false，满了选不了药水
        if "choose" in game_state_obj.available_commands:
            # 判断当前是否是商店页面，且药水栏满了而且钱够买药水
            if game_state_obj.screen_type == ScreenType.SHOP_SCREEN and game_state_obj.are_potions_full():
                # 收集所有需要屏蔽掉的name (使用 set 加速查找)
                potion_names = set()
                if hasattr(game_state_obj.screen, 'potions'):
                    for potion in game_state_obj.screen.potions:
                        potion_names.add(potion.name)
                
                # 遍历 choice_list，如果名字在 potion_names 中，则屏蔽
                # 这种方式可以正确处理商店中有多个同名药水的情况
                if hasattr(game_state_obj, 'choice_list'):
                    for idx, choice_name in enumerate(game_state_obj.choice_list):
                        if choice_name in potion_names:
                            masks['choose_option'][idx] = 0

                # 如果除了药水之外没有别的选项了，就把choose_option全屏蔽
                choose_mask = masks['choose_option']
                # np底层优化过的函数，判断非零元素个数，比sum快 且更准确
                if np.count_nonzero(choose_mask) == 0: 
                    masks['action_type'][DecomposedActionType.CHOOSE.value] = 0

            # potion_idxs = self.choose_index_based_name(game_state.choice_list, 'potion')
            potion_idxs = [i for i, choice in enumerate(game_state_obj.choice_list) if choice == 'potion'] if hasattr(game_state_obj, 'choice_list') else []
            
            if potion_idxs and game_state_obj.are_potions_full():
                # 可能同时有好几个药水选项
                for potion_idx in potion_idxs:
                    masks['choose_option'][potion_idx] = 0  # 不能选药水了
                # 如果除了药水之外没有别的选项了，就把choose_option全屏蔽
                choose_mask = masks['choose_option']
                # np底层优化过的函数，判断非零元素个数，比sum快 且更准确
                if np.count_nonzero(choose_mask) == 0: 
                    masks['action_type'][DecomposedActionType.CHOOSE.value] = 0

        # 应用 Mask
        action_type_mask = torch.from_numpy(masks['action_type']).bool().to(self.device)
        
        # --- 安全检查：防止所有动作都被 Mask 掉导致 Crash ---
        if not action_type_mask.any():
            # 尝试从 available_commands 恢复
            if hasattr(game_state_obj, 'available_commands'):
                cmds = game_state_obj.available_commands
                if 'end' in cmds:
                    action_type_mask[DecomposedActionType.END.value] = True
                elif 'proceed' in cmds:
                    action_type_mask[DecomposedActionType.PROCEED.value] = True
                elif 'confirm' in cmds:
                    action_type_mask[DecomposedActionType.CONFIRM.value] = True
                elif 'skip' in cmds:
                    action_type_mask[DecomposedActionType.SKIP.value] = True
                elif 'leave' in cmds:
                    action_type_mask[DecomposedActionType.LEAVE.value] = True
                elif 'return' in cmds:
                    action_type_mask[DecomposedActionType.RETURN.value] = True
            
            # 如果还是全 False (比如 available_commands 解析失败)，强制开启 END 以防 Crash
            if not action_type_mask.any():
                # print("Warning: All actions masked! Forcing END.")
                action_type_mask[DecomposedActionType.END.value] = True

        q_type[~action_type_mask] = -float('inf')
        
        # --- 动作类型选择 ---
        if self.is_training:
            probs = torch.softmax(q_type / self.temperature, dim=-1)
            # 二次检查：防止 probs 含有 NaN (例如 q_type 全是 -inf)
            if torch.isnan(probs).any() or probs.sum() == 0:
                # 均匀分布回退
                probs = torch.zeros_like(probs)
                probs[action_type_mask] = 1.0 / action_type_mask.sum().float()
            
            act_idx = torch.multinomial(probs, 1).item()
        else:
            act_idx = torch.argmax(q_type).item()
            
        action_type = DecomposedActionType(act_idx)
        
        # --- 分支参数选择 ---
        # 辅助函数: 根据温度采样
        def select_idx(q_vals, mask_key):
            mask = torch.from_numpy(masks[mask_key]).bool().to(self.device)
            # 截断 mask 以匹配 q_vals 长度 (防止越界)
            if len(mask) > len(q_vals):
                mask = mask[:len(q_vals)]
            elif len(mask) < len(q_vals):
                # Pad mask with False
                padding = torch.zeros(len(q_vals) - len(mask), dtype=torch.bool, device=self.device)
                mask = torch.cat([mask, padding])
                
            q_vals[~mask] = -float('inf')
            
            if self.is_training:
                p = torch.softmax(q_vals / self.temperature, dim=-1)
                return torch.multinomial(p, 1).item()
            else:
                return torch.argmax(q_vals).item()

        # 根据类型组装 Action 对象
        if action_type == DecomposedActionType.PLAY:
            hand_idx = select_idx(q_card, 'play_card')
            
            target_idx = None
            # 判断是否需要目标
            card = game_state_obj.hand[hand_idx]
            if card.has_target:
                target_idx = select_idx(q_monster, 'target_monster')
                
            return PlayAction(ActionType.PLAY, hand_idx, target_idx)
            
        elif action_type == DecomposedActionType.CHOOSE:
            # 检查是否是 Shop 进入逻辑
            if game_state_obj.screen_type == ScreenType.SHOP_ROOM:
                # 站在商店门口
                # ... (简化逻辑，直接选)
                pass
            
            choice_idx = select_idx(q_choice, 'choose_option')
            return ChooseAction(ActionType.CHOOSE, choice_idx)
            
        elif action_type == DecomposedActionType.POTION_USE:
            pot_idx = select_idx(q_pot_use, 'potion_use')
            target_idx = None
            if game_state_obj.potions[pot_idx].requires_target:
                target_idx = select_idx(q_monster, 'target_monster')
            return PotionUseAction(ActionType.POTION_USE, pot_idx, target_idx)
            
        elif action_type == DecomposedActionType.POTION_DISCARD:
            pot_idx = select_idx(q_pot_disc, 'potion_discard')
            return PotionDiscardAction(ActionType.POTION_DISCARD, pot_idx)
            
        else:
            # End, Proceed, etc.
            base_type = action_type.to_action_type()
            from spirecomm.ai.dqn_core.action import SingleAction
            return SingleAction(base_type, decomposed_type=action_type)

# class DQN:
#     def __init__(self, state_size, state_processor:GameStateProcessor):
#         self.state_size = state_size
#         self.memory = deque(maxlen=2000) # 经验回放池，使用固定大小的双端队列
#         self.gamma = 0.95    # 折扣因子
#         # --- Boltzmann 探索参数 ---
#         self.temperature = 5.0  # 初始温度，高温度意味着更随机的探索
#         self.temperature_min = 0.1 # 最低温度
#         self.temperature_decay = 0.999 # 温度衰减率
#         # -------------------------
#         self.is_training = True # 默认为训练模式
#         self.state_processor = state_processor
        
#         self.visited_shop = False  # 用于跟踪是否已经访问过商店

#         # --- 神经网络 ---
#         # 策略网络 (Policy Network): 用于决定下一步动作，我们会频繁更新它
#         self.policy_net = DQNModel(state_size)
#         # 目标网络 (Target Network): 用于计算目标Q值，它的权重是定期从策略网络复制过来的，用于稳定训练
#         self.target_net = DQNModel(state_size)
#         self.target_net.load_state_dict(self.policy_net.state_dict()) # 初始化时，权重完全相同
#         self.target_net.eval() # 目标网络只用于推理，不进行训练
        
#         self.optimizer = optim.Adam(self.policy_net.parameters())
#         self.loss_fn = nn.MSELoss()

#         self.qvalue_logger = AbsoluteLogger(log_type=LogType.QVALUE)
#         self.qvalue_logger.start_episode()
        
#     def remember(self, state, action, reward, next_state, done):
#         state_numpy = state.detach().cpu().numpy()
#         next_state_numpy = next_state.detach().cpu().numpy()
#         self.memory.append((state_numpy, action, reward, next_state_numpy, done))

#     def train(self, batch_size=32):
#         # 如果经验池中的样本不足一个批次，则不进行训练
#         if len(self.memory) < batch_size:
#             return
#         # 从经验回放池中随机采样一个批次
#         minibatch = random.sample(self.memory, batch_size)
        
#         # 将经验元组解压并转换为PyTorch张量
#         states = torch.from_numpy(np.vstack([e[0] for e in minibatch if e is not None])).float()
#         actions = [e[1] for e in minibatch if e is not None]
#         rewards = torch.tensor([e[2] for e in minibatch if e is not None]).float()
#         next_states = torch.from_numpy(np.vstack([e[3] for e in minibatch if e is not None])).float()
#         dones = torch.tensor([e[4] for e in minibatch if e is not None]).float()
#         # 在这里断言以上张量的形状正确（强断言，便于尽早定位问题）
#         batch = states.shape[0]
#         assert batch == len(actions) == rewards.shape[0] == next_states.shape[0] == dones.shape[0], \
#             f"batch size mismatch: states {states.shape[0]}, actions {len(actions)}, rewards {rewards.shape[0]}, next_states {next_states.shape[0]}, dones {dones.shape[0]}"
#         assert states.dim() == 2 and states.shape[1] == self.state_size, \
#             f"state vector size mismatch: expected {self.state_size}, got {states.shape[1]}"
#         # 确保 dtype/device 可接受
#         assert states.dtype.is_floating_point and next_states.dtype.is_floating_point, "state tensors must be float"
#         # 记录当前采样批次信息（便于调试）

#         # --- 1. 计算预测Q值 (Predicted Q-values) ---
#         # 使用策略网络(policy_net)获取当前状态的Q值
#         pred_action_q, pred_arg_q = self.policy_net(states)
#         expected_action_types = len(DecomposedActionType)
#         assert pred_action_q.dim() == 2 and pred_action_q.shape[0] == batch and pred_action_q.shape[1] == expected_action_types, \
#             f"pred_action_q shape unexpected: got {tuple(pred_action_q.shape)}, expected (batch, {expected_action_types})"
#         # 检查参数头存在并形状正确
#         assert isinstance(pred_arg_q, dict), "pred_arg_q must be a dict of argument heads"
#         assert 'play_card' in pred_arg_q and pred_arg_q['play_card'].dim() == 2 and pred_arg_q['play_card'].shape[0] == batch and pred_arg_q['play_card'].shape[1] == MAX_HAND_SIZE, \
#             f"play_card head shape incorrect: {pred_arg_q.get('play_card').shape if 'play_card' in pred_arg_q else None}"
#         assert 'target_monster' in pred_arg_q and pred_arg_q['target_monster'].dim() == 2 and pred_arg_q['target_monster'].shape[0] == batch and pred_arg_q['target_monster'].shape[1] == MAX_MONSTER_COUNT, \
#             f"target_monster head shape incorrect: {pred_arg_q.get('target_monster').shape if 'target_monster' in pred_arg_q else None}"
#         assert 'choose_option' in pred_arg_q and pred_arg_q['choose_option'].dim() == 2 and pred_arg_q['choose_option'].shape[0] == batch and pred_arg_q['choose_option'].shape[1] == MAX_DECK_SIZE, \
#             f"choose_option head shape incorrect: {pred_arg_q.get('choose_option').shape if 'choose_option' in pred_arg_q else None}"
#         # 现在药水分为 use / discard 两个独立头
#         assert 'potion_use' in pred_arg_q and pred_arg_q['potion_use'].dim() == 2 and pred_arg_q['potion_use'].shape[0] == batch and pred_arg_q['potion_use'].shape[1] == MAX_POTION_COUNT, \
#             f"potion_use head shape incorrect: {pred_arg_q.get('potion_use').shape if 'potion_use' in pred_arg_q else None}"
#         assert 'potion_discard' in pred_arg_q and pred_arg_q['potion_discard'].dim() == 2 and pred_arg_q['potion_discard'].shape[0] == batch and pred_arg_q['potion_discard'].shape[1] == MAX_POTION_COUNT, \
#             f"potion_discard head shape incorrect: {pred_arg_q.get('potion_discard').shape if 'potion_discard' in pred_arg_q else None}"

#         # 逐样本提取标量 Q 值并保留计算图（不转为 Python float）
#         predicted_q_values_list = []
#         debug_lines = []
#         for i, action in enumerate(actions):
#             # 保证 action 是我们定义的结构体
#             assert action is not None, f"action at idx {i} is None"
#             action_idx = self._action_index_from_decomposed(action)
#              # 容错：确保 action_idx 是 int（理论上 _action_index_from_decomposed 已保证，但加二重保险）
#             if action_idx is None:
#                 debug_lines.append(f"action_idx is None for batch_idx={i}, action={repr(action)}; fallback to 0")
#                 action_idx = 0
#             try:
#                 action_idx = int(action_idx)
#             except Exception as e:
#                 debug_lines.append(f"failed to convert action_idx to int for batch_idx={i}, action={repr(action)}: {e}; fallback to 0")
#                 action_idx = 0
#             # 主动作Q（确保 action_idx 为 int 且在范围内）
#             assert 0 <= action_idx < expected_action_types, f"action_idx out of range: {action_idx} (expected 0..{expected_action_types-1})"
#             q_val = pred_action_q[i, action_idx]

#             # 参数Q累加（索引均强制为 int）
#             if isinstance(action, PlayAction):
#                 # 确保 hand_idx/target_idx 是 int 且在合法范围
#                 hand_idx = int(action.hand_idx)
#                 assert 0 <= hand_idx < MAX_HAND_SIZE, f"hand_idx out of range: {hand_idx}"
#                 q_val = q_val + pred_arg_q['play_card'][i, hand_idx]
#                 if action.target_idx is not None:
#                     target_idx = int(action.target_idx)
#                     assert 0 <= target_idx < MAX_MONSTER_COUNT, f"target_idx out of range: {target_idx}"
#                     q_val = q_val + pred_arg_q['target_monster'][i, target_idx]
#             elif isinstance(action, ChooseAction):
#                 choice_idx = int(action.choice_idx)
#                 assert 0 <= choice_idx < MAX_DECK_SIZE, f"choice_idx out of range: {choice_idx}"
#                 q_val = q_val + pred_arg_q['choose_option'][i, choice_idx]
#             elif isinstance(action, PotionUseAction):
#                 potion_idx = int(action.potion_idx)
#                 assert 0 <= potion_idx < MAX_POTION_COUNT, f"potion_idx out of range: {potion_idx}"
#                 q_val = q_val + pred_arg_q['potion_use'][i, potion_idx]
#                 if action.target_idx is not None:
#                     target_idx = int(action.target_idx)
#                     assert 0 <= target_idx < MAX_MONSTER_COUNT, f"target_idx out of range: {target_idx}"
#                     q_val = q_val + pred_arg_q['target_monster'][i, target_idx]
#             elif isinstance(action, PotionDiscardAction):
#                 potion_idx = int(action.potion_idx)
#                 assert 0 <= potion_idx < MAX_POTION_COUNT, f"potion_idx out of range: {potion_idx}"
#                 q_val = q_val + pred_arg_q['potion_discard'][i, potion_idx]

#             # 保持为 torch scalar（0-d tensor）以保留梯度；若 q_val 非标量则 squeeze 或取第一个元素
#             if isinstance(q_val, torch.Tensor):
#                 if q_val.numel() == 0:
#                     # 空张量回退为 0
#                     q_val = torch.tensor(0.0, device=q_val.device, dtype=q_val.dtype)
#                 elif q_val.numel() > 1:
#                     # 多元素张量取均值（或首元素），保留计算图
#                     q_val = q_val.reshape(-1).mean()
#                 # q_val.numel()==1 时保持原样（可能是 0-d 或 [1] 的 1-d tensor）
#             predicted_q_values_list.append(q_val)

#         # 写调试日志（若有）
#         if debug_lines:
#             try:
#                 with open("training_debug.log", "a", encoding="utf-8") as df:
#                     for ln in debug_lines:
#                         df.write(ln + "\n")
#             except Exception:
#                 pass
#         # 用 stack 组装为 1D tensor（保留梯度）
#         predicted_q_values = torch.stack(predicted_q_values_list)
#         assert predicted_q_values.dim() == 1 and predicted_q_values.shape[0] == batch, \
#             f"predicted_q_values shape invalid: {tuple(predicted_q_values.shape)}, expected ({batch},)"

#         # --- 2. 计算目标Q值 (Target Q-values) ---
#         # 使用目标网络(target_net)来计算下一状态的最大Q值，这可以稳定训练过程
#         with torch.no_grad():
#             next_action_q, next_arg_q = self.target_net(next_states)
#             # 找到下一状态中，Q值最高的合法动作类型
#             # 注意：这里为了简化，我们只考虑了下一状态的动作类型Q值，
#             # TODO 可能更精确的方法更好，但我觉得估计这样应该就行了
#             # 一个更精确的方法是找到下一状态Q值最高的完整动作（类型+参数）
#             max_next_q = next_action_q.max(1)[0]
#         assert max_next_q.dim() == 1 and max_next_q.shape[0] == batch, \
#             f"max_next_q shape invalid: {tuple(max_next_q.shape)}, expected ({batch},)"

#         # 贝尔曼方程: Target = reward + gamma * max_next_q
#         # 如果是回合结束(done=True)，则没有未来奖励，Target = reward
#         target_q_values = rewards + (1 - dones) * self.gamma * max_next_q
#         assert target_q_values.shape == (batch,), f"target_q_values shape mismatch: {tuple(target_q_values.shape)} vs ({batch},)"

#          # --- 3. 计算损失并进行反向传播 ---
#          # 此时 predicted_q_values 和 target_q_values 都应为 shape=(batch,)
#         loss = self.loss_fn(predicted_q_values, target_q_values)
        
#         # 输出所有 Q 值，loss 到日志，便于分析
#         try:
#             all_q_values = pred_action_q.detach().cpu().numpy()
#             self.qvalue_logger.write({
#                 "loss": loss.item(),
#                 "predicted_q_values": predicted_q_values.detach().cpu().numpy().tolist(),
#                 "target_q_values": target_q_values.detach().cpu().numpy().tolist(),
#                 "all_action_q_values": all_q_values.tolist()
#             })
            
#             if self.wandb_logger:
#                 self.wandb_logger.log_metrics({
#                     "train/loss": loss.item(),
#                     "train/avg_q": predicted_q_values.mean().item(),
#                     "train/epsilon": self.temperature # Assuming temperature is used for exploration
#                 })
#         except Exception:
#             pass
#         # 梯度清零
#         self.optimizer.zero_grad()
#         # 反向传播
#         loss.backward()
#         # 更新策略网络的权重
#         self.optimizer.step()

#         # 在每次训练后衰减温度
#         if self.temperature > self.temperature_min:
#             self.temperature *= self.temperature_decay

#     def set_inference_mode(self):
#         """切换到推理模式，不进行探索。"""
#         self.is_training = False

#     def choose_action(self, state_tensor, masks, game_state:Game):
#         """
#         使用 Boltzmann 探索 (Softmax 探索) 来选择一个分解式动作。
#         Q值越高的动作被选择的概率越大。
#         :param state_tensor: 当前状态的 PyTorch 张量。
#         :param masks: 一个包含所有合法动作掩码的字典。
#         :return: 一个结构化的动作对象 (e.g., PlayAction, ChooseAction)。
#         """
#         action_type_q, arg_q = self.get_q_values(state_tensor, use_policy_net=True)

#         # --- Wandb Logging Prep ---
#         if self.wandb_logger:
#             try:
#                 self.last_q_values = {}
                
#                 # 1. Action Types
#                 at_q = action_type_q.squeeze(0).detach().cpu().numpy()
#                 at_mask = masks.get('action_type', np.ones_like(at_q, dtype=bool))
#                 for i, val in enumerate(at_q):
#                     if i < len(at_mask) and at_mask[i]:
#                         try:
#                             name = DecomposedActionType(i).name
#                         except:
#                             name = f"Type_{i}"
#                         self.last_q_values[name] = float(val)

#                 # 2. Arguments
#                 # arg_q is a dict of tensors [Batch, Size]
                
#                 # Play Card
#                 if 'play_card' in arg_q:
#                     pc_q = arg_q['play_card'].squeeze(0).detach().cpu().numpy()
#                     pc_mask = masks.get('play_card', np.zeros_like(pc_q, dtype=bool))
#                     for i, val in enumerate(pc_q):
#                         if i < len(pc_mask) and pc_mask[i]:
#                             card_name = f"Card_{i}"
#                             if i < len(game_state.hand):
#                                 card_name = f"Card:{game_state.hand[i].name}"
#                             self.last_q_values[card_name] = float(val)

#                 # Target Monster
#                 if 'target_monster' in arg_q:
#                     tm_q = arg_q['target_monster'].squeeze(0).detach().cpu().numpy()
#                     tm_mask = masks.get('target_monster', np.zeros_like(tm_q, dtype=bool))
#                     for i, val in enumerate(tm_q):
#                         if i < len(tm_mask) and tm_mask[i]:
#                             mon_name = f"Monster_{i}"
#                             if i < len(game_state.monsters):
#                                 mon_name = f"Monster:{game_state.monsters[i].name}"
#                             self.last_q_values[mon_name] = float(val)

#                 # Choose Option
#                 if 'choose_option' in arg_q:
#                     co_q = arg_q['choose_option'].squeeze(0).detach().cpu().numpy()
#                     co_mask = masks.get('choose_option', np.zeros_like(co_q, dtype=bool))
#                     for i, val in enumerate(co_q):
#                         if i < len(co_mask) and co_mask[i]:
#                             opt_name = f"Choice_{i}"
#                             if hasattr(game_state, 'choice_list') and i < len(game_state.choice_list):
#                                 opt_name = f"Choice:{game_state.choice_list[i]}"
#                             self.last_q_values[opt_name] = float(val)

#                 # Potion Use
#                 if 'potion_use' in arg_q:
#                     pu_q = arg_q['potion_use'].squeeze(0).detach().cpu().numpy()
#                     pu_mask = masks.get('potion_use', np.zeros_like(pu_q, dtype=bool))
#                     for i, val in enumerate(pu_q):
#                         if i < len(pu_mask) and pu_mask[i]:
#                             pot_name = f"PotUse_{i}"
#                             if i < len(game_state.potions):
#                                 pot_name = f"PotUse:{game_state.potions[i].name}"
#                             self.last_q_values[pot_name] = float(val)

#                 # Potion Discard
#                 if 'potion_discard' in arg_q:
#                     pd_q = arg_q['potion_discard'].squeeze(0).detach().cpu().numpy()
#                     pd_mask = masks.get('potion_discard', np.zeros_like(pd_q, dtype=bool))
#                     for i, val in enumerate(pd_q):
#                         if i < len(pd_mask) and pd_mask[i]:
#                             pot_name = f"PotDisc_{i}"
#                             if i < len(game_state.potions):
#                                 pot_name = f"PotDisc:{game_state.potions[i].name}"
#                             self.last_q_values[pot_name] = float(val)

#             except Exception as e:
#                 self.last_q_values = {}

#         # 1. 决策第一步：选择动作类型 (Action Type)
#         action_type_q = action_type_q.squeeze(0) # 移除 batch 维度
#         action_type_mask = torch.from_numpy(masks['action_type']).bool()
        
#         # 应用掩码，将非法动作的Q值设为负无穷
#         action_type_q[~action_type_mask] = -float('inf')
        
#         # 如果所有动作类型都被屏蔽了
#         if not action_type_mask.any():
#             # 这种情况通常不应该发生，但为了防止崩溃，返回 END 或 PROCEED
#             from spirecomm.ai.dqn_core.action import SingleAction
#             if 'end' in game_state.available_commands:
#                 return SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END)
#             elif 'proceed' in game_state.available_commands or 'confirm' in game_state.available_commands:
#                 return SingleAction(type=ActionType.PROCEED, decomposed_type=DecomposedActionType.PROCEED)
#             else:
#                 # 实在没办法，随便返回一个 END，让游戏逻辑去处理无效操作
#                 return SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END)
        
#         # 判断药水是否满了
#         # 如果状态里面的choice_list有potion字段的话，把对应的index选出来，mask置为false，满了选不了药水
#         if "choose" in game_state.available_commands:
#             # 判断当前是否是商店页面，且药水栏满了而且钱够买药水
#             if game_state.screen_type == ScreenType.SHOP_SCREEN and game_state.are_potions_full():
#                 # 收集所有需要屏蔽掉的name
#                 potion_names = [potion.name for potion in game_state.screen.potions]
#                 # 获得 potion_names 中每一个 name 在 choice_list 对应的 index（安全地跳过不存在的名字）
#                 potion_idxs = []
#                 for name in potion_names:
#                     try:
#                         potion_idxs.append(game_state.choice_list.index(name))
#                     except ValueError:
#                         # 如果某个名字不在 choice_list 中，可能表示不可购买/已经不存在，直接跳过
#                         continue
#                  # 可能同时有好几个药水选项
#                 if potion_idxs:
#                     for potion_idx in potion_idxs:
#                         masks['choose_option'][potion_idx] = 0  # 不能选药水了
#                 # 如果除了药水之外没有别的选项了，就把choose_option全屏蔽
#                 choose_mask:np.ndarray = masks['choose_option']
#                 # np底层优化过的函数，判断非零元素个数，比sum快 且更准确
#                 if np.count_nonzero(choose_mask) == 0: 
#                     action_type_q[DecomposedActionType.CHOOSE.value] = -float('inf')

#             potion_idxs = self.choose_index_based_name(game_state.choice_list, 'potion')
#             if potion_idxs is not None and game_state.are_potions_full():
#                 # 可能同时有好几个药水选项
#                 for potion_idx in potion_idxs:
#                     masks['choose_option'][potion_idx] = 0  # 不能选药水了
#                 # 如果除了药水之外没有别的选项了，就把choose_option全屏蔽
#                 choose_mask:np.ndarray = masks['choose_option']
#                 # np底层优化过的函数，判断非零元素个数，比sum快 且更准确
#                 if np.count_nonzero(choose_mask) == 0: 
#                     action_type_q[DecomposedActionType.CHOOSE.value] = -float('inf')
#         if self.is_training:
#             # 训练模式：Boltzmann 探索
#             action_type_probs = torch.softmax(action_type_q / self.temperature, dim=-1)
#             # 从概率分布中采样一个动作类型
#             action_type_idx = torch.multinomial(action_type_probs, 1).item()
#         else:
#             # 推理模式：选择Q值最高的动作
#             action_type_idx = torch.argmax(action_type_q).item()

#         action_type = DecomposedActionType(action_type_idx)

#         # 2. 决策第二步：根据动作类型选择参数
#         if action_type == DecomposedActionType.PLAY:
#             # 需要选择打哪张牌 (play_card) 和目标 (target_monster)
#             play_card_q = arg_q['play_card'].squeeze(0)
#             play_card_mask = torch.from_numpy(masks['play_card']).bool()
            
#             # 安全检查：如果 mask 全为 False，说明没有牌可打，回退到 END
#             if not play_card_mask.any():
#                 from spirecomm.ai.dqn_core.action import SingleAction
#                 return SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END)

#             play_card_q[~play_card_mask] = -float('inf')
#             if self.is_training:
#                 play_card_probs = torch.softmax(play_card_q / self.temperature, dim=-1)
#                 card_idx = torch.multinomial(play_card_probs, 1).item()
#             else:
#                 card_idx = torch.argmax(play_card_q).item()

#             # 选中了牌之后，判断这个牌是否需要目标
#             target_idx = None
#             chosen_card:Card = game_state.hand[card_idx]
#             if chosen_card.has_target:
#                 target_q = arg_q['target_monster'].squeeze(0)
#                 target_mask = torch.from_numpy(masks['target_monster']).bool()
                
#                 # 安全检查：如果需要目标但没有合法目标
#                 if not target_mask.any():
#                      # 这种情况很罕见（有目标牌但无目标怪？），可能怪全死了？
#                      # 简单处理：不选目标，或者回退
#                      target_idx = None
#                 else:
#                     target_q[~target_mask] = -float('inf')
#                     if self.is_training:
#                         target_probs = torch.softmax(target_q / self.temperature, dim=-1)
#                         target_idx = torch.multinomial(target_probs, 1).item()
#                     else:
#                         target_idx = torch.argmax(target_q).item()

#             # 如果目标掩码全为False，说明该牌无需目标
#             if not np.any(masks['target_monster']):
#                 target_idx = None

#             return PlayAction(type=ActionType.PLAY, hand_idx=card_idx, target_idx=target_idx)

#         elif action_type == DecomposedActionType.CHOOSE:
#             # 当遇到商店选项的时候，首先要看需不需要进入商店
#             if game_state.screen_type == ScreenType.SHOP_ROOM:
#                 # 站在商店门口
#                 if not self.visited_shop:
#                     self.visited_shop = True
#                     for i, option in enumerate(game_state.choice_list):
#                         if option == 'shop':
#                             # 没来过的话就进去看看
#                             return ChooseAction(type=ActionType.CHOOSE, choice_idx=i)
#                 else:
#                     # 来过了的话，proceed 继续，然后后续必须立马往前进 TODO，否则再选到return的话，就在商店门口死循环了。
#                     self.visited_shop = False
#                     from spirecomm.ai.dqn_core.action import SingleAction
#                     return SingleAction(type=ActionType.PROCEED, decomposed_type=ActionType.PROCEED)
            
#             # 安全检查：确保 choice_list 存在且不为空
#             if not hasattr(game_state, 'choice_list') or not game_state.choice_list:
#                 from spirecomm.ai.dqn_core.action import SingleAction
#                 return SingleAction(type=ActionType.PROCEED, decomposed_type=DecomposedActionType.PROCEED)

#             choose_q = arg_q['choose_option'].squeeze(0)
#             choose_mask = torch.from_numpy(masks['choose_option']).bool()
            
#             # 安全检查
#             if not choose_mask.any():
#                  from spirecomm.ai.dqn_core.action import SingleAction
#                  return SingleAction(type=ActionType.PROCEED, decomposed_type=DecomposedActionType.PROCEED)

#             choose_q[~choose_mask] = -float('inf')
#             if self.is_training:
#                 choose_probs = torch.softmax(choose_q / self.temperature, dim=-1)
#                 choice_idx = torch.multinomial(choose_probs, 1).item()
#             else:
#                 choice_idx = torch.argmax(choose_q).item()
#             return ChooseAction(type=ActionType.CHOOSE, choice_idx=choice_idx)

#         elif action_type == DecomposedActionType.POTION_USE:
#             # 需要选择使用哪个药水，以及可能的目标
#             # 使用专门的 potion_use 头与掩码；向后兼容仍可接受旧的 'potion'
#             potion_q = arg_q.get('potion_use').squeeze(0)
#             potion_mask = torch.from_numpy(masks.get('potion_use')).bool()
            
#             # 安全检查
#             if not potion_mask.any():
#                  from spirecomm.ai.dqn_core.action import SingleAction
#                  return SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END)

#             potion_q[~potion_mask] = -float('inf')
#             if self.is_training:
#                 potion_probs = torch.softmax(potion_q / self.temperature, dim=-1)
#                 potion_idx = torch.multinomial(potion_probs, 1).item()
#             else:
#                 potion_idx = torch.argmax(potion_q).item()

#             target_idx = None
#             # 检查使用该药水是否需要目标
#             chosen_potion:Potion = game_state.potions[potion_idx]
#             if chosen_potion.requires_target:
#                 target_q = arg_q['target_monster'].squeeze(0)
#                 target_mask = torch.from_numpy(masks['target_monster']).bool()
                
#                 if not target_mask.any():
#                     target_idx = None
#                 else:
#                     target_q[~target_mask] = -float('inf')
#                     if self.is_training:
#                         target_probs = torch.softmax(target_q / self.temperature, dim=-1)
#                         target_idx = torch.multinomial(target_probs, 1).item()
#                     else:
#                         target_idx = torch.argmax(target_q).item()
            
#             return PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=target_idx)
        
#         elif action_type == DecomposedActionType.POTION_DISCARD:
#             # 需要选择丢弃哪个药水
#             potion_q = arg_q['potion_discard'].squeeze(0)
#             potion_mask = torch.from_numpy(masks.get('potion_discard')).bool()
            
#             # 安全检查
#             if not potion_mask.any():
#                  from spirecomm.ai.dqn_core.action import SingleAction
#                  return SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END)

#             potion_q[~potion_mask] = -float('inf')
#             if self.is_training:
#                 potion_probs = torch.softmax(potion_q / self.temperature, dim=-1)
#                 potion_idx = torch.multinomial(potion_probs, 1).item()
#             else:
#                 potion_idx = torch.argmax(potion_q).item()
#             return PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx)
            
#         # 对于所有无参数的动作 (END, PROCEED, CANCEL, LEAVE, CONFIRM)
#         else:
#             # 这些动作没有参数，直接构建并返回SingleAction对象
#             # 我们需要一个方法将DecomposedActionType映射回ActionType
#             base_action_type = action_type.to_action_type()
#             if base_action_type is None:
#                 raise ValueError(f"无法将有参数的分解动作 {action_type.name} 转换为 SingleAction")
#             return SingleAction(type=base_action_type, decomposed_type=action_type)

#     def choose_index_based_name(self, choice_list, name):
#         """根据名称选择对应的索引"""
#         all_indices = [i for i, choice in enumerate(choice_list) if choice == name]
#         return all_indices
#     def get_q_values(self, state, use_policy_net=True):
#         """获取所有头的Q值"""
#         with torch.no_grad():
#             if use_policy_net:
#                 self.policy_net.eval()
#                 action_q, arg_q = self.policy_net(state)
#                 if self.is_training: # 只有在训练模式下才切换回 .train()
#                     self.policy_net.train()
#             else: # 使用目标网络
#                 action_q, arg_q = self.target_net(state)
#         return action_q, arg_q

#     def get_all_legal_action_q_values(self, state_tensor, game_state):
#         """
#         获取当前状态下所有合法动作的Q值，用于日志记录。
#         :param state_tensor: 当前状态的 PyTorch 张量。
#         :param game_state: 原始游戏状态字典，用于获取合法动作。
#         :return: 一个字典，键是动作字符串，值是对应的Q值。
#         """
#         q_values = {}
#         available_commands = self.state_processor.get_available_actions(game_state)
#         if not available_commands:
#             return q_values

#         # 使用策略网络获取Q值
#         action_type_q, arg_q = self.get_q_values(state_tensor, use_policy_net=True)
#         action_type_q = action_type_q.squeeze(0)
#         arg_q = {k: v.squeeze(0) for k, v in arg_q.items()}

#         for action in available_commands:
#             # 分解式动作没有直接的Q值，我们需要根据其构成来计算
#             # 注意：这里的计算方式和训练时的Q值计算方式保持一致
#             q_val = 0
#             # 动作类型的Q值
#             if hasattr(action, 'decomposed_type'):
#                 q_val += action_type_q[action.decomposed_type.value].item()
            
#             # 参数的Q值
#             if isinstance(action, PlayAction):
#                 q_val += arg_q['play_card'][action.hand_idx].item()
#                 if action.target_idx is not None:
#                     q_val += arg_q['target_monster'][action.target_idx].item()
#             elif isinstance(action, ChooseAction):
#                 q_val += arg_q['choose_option'][action.choice_idx].item()
#             elif isinstance(action, PotionUseAction):
#                 q_val += arg_q['potion_use'][action.potion_idx].item()
#                 if action.target_idx is not None:
#                     q_val += arg_q['target_monster'][action.target_idx].item()
#             elif isinstance(action, PotionDiscardAction):
#                 q_val += arg_q['potion_discard'][action.potion_idx].item()
            
#             q_values[action.to_string()] = q_val
#         return q_values

#     def update_target_net(self):
#         """定期将策略网络的权重复制到目标网络"""
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#     # 新增：把 action.decomposed_type 安全地转换为整数索引
#     def _action_index_from_decomposed(self, action):
#         """
#         返回可用于索引 pred_action_q 的整数索引。
#         兼容场景：
#           - action.decomposed_type.value 是 int
#           - action.decomposed_type.value 是可以转为 int 的字符串
#           - action.decomposed_type 是 Enum/IntEnum（通过成员的 value 或顺序）
#           - 如果实例上定义了 ACTION_INDEX_MAP / action_index_map 字典，则使用该映射（支持 key 为 name 或 value）
#           失败时回退为 0（确保返回值永远是 int）。
#         """
#         # 尝试取 value
#         dt = getattr(action, "decomposed_type", None)
#         if dt is None:
#             # action 没有 decomposed_type 属性，回退为 0
#             return 0
#         val = getattr(dt, "value", None)

#         # 直接是整数
#         if isinstance(val, int):
#             return int(val)
#         # value 为可转 int 的字符串
#         if isinstance(val, str):
#             try:
#                 return int(val)
#             except Exception:
#                 pass
#         # 如果 decomposed_type 本身是 int 或可转为 int
#         if isinstance(dt, int):
#             return int(dt)
#         if isinstance(dt, str):
#             try:
#                 return int(dt)
#             except Exception:
#                 pass
#         # 如果是 Enum/IntEnum，尝试用成员的 value 或成员顺序
#         try:
#             if hasattr(dt, "value"):
#                 member_val = dt.value
#                 if isinstance(member_val, int):
#                     return int(member_val)
#             enum_type = type(dt)
#             if hasattr(enum_type, "__members__"):
#                 try:
#                     members = list(enum_type)
#                     return members.index(dt)
#                 except Exception:
#                     pass
#         except Exception:
#             pass
#         # 尝试类实例上的自定义映射
#         for attr in ("ACTION_INDEX_MAP", "action_index_map"):
#             m = getattr(self, attr, None)
#             if isinstance(m, dict):
#                 # 尝试按 name 或 value 或字符串键匹配
#                 key_candidates = []
#                 if hasattr(dt, "name"):
#                     key_candidates.append(dt.name)
#                 if isinstance(val, str):
#                     key_candidates.append(val)
#                 key_candidates.append(str(dt))
#                 for k in key_candidates:
#                     if k in m:
#                         try:
#                             return int(m[k])
#                         except Exception:
#                             continue
#         # 最后回退到 0（确保返回值是 int）
#         return 0