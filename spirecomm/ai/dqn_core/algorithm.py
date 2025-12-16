from calendar import c
from dataclasses import fields
import random
from collections import deque
from re import purge
import sys
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
        self.temperature_min = 0.1

        self.temperature_start = 1.77
        self.temperature = self.temperature_start
        self.exploration_total_steps = 400000  # 计划的总探索步数
        self.temperature_decay = 0.99999
        self.is_training = True
        self.absolute_logger = AbsoluteLogger(LogType.STATE)
        self.absolute_logger.start_episode()
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
        # --- 关键修复：在 loss 前过滤非有限值，避免污染训练 ---
        finite_mask = torch.isfinite(pred_q_tensor) & torch.isfinite(target_q_tensor)
        dropped = int((~finite_mask).sum().item())

        if finite_mask.any():
            loss = self.loss_fn(pred_q_tensor[finite_mask], target_q_tensor[finite_mask])
        else:
            # 整个 batch 都坏了：构造一个 0 loss，跳过反传
            loss = torch.zeros((), device=self.device, dtype=torch.float32)

        # 只有 loss 有限且有有效样本时才更新参数
        did_step = False
        if finite_mask.any() and torch.isfinite(loss):
            self.optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸 (Transformer/LSTM常用)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            did_step = True

        self.total_steps += 1

        if self.wandb_logger:
            # 统计 pred/target 内的数值健康度（用于定位是否只是 log 问题）
            pred_finite = torch.isfinite(pred_q_tensor)
            targ_finite = torch.isfinite(target_q_tensor)
            pred_bad = int((~pred_finite).sum().item())
            targ_bad = int((~targ_finite).sum().item())

            # 过滤掉可能存在的 -inf / inf / nan，避免污染 Log 统计
            valid_q = pred_q_tensor[pred_finite]
            if valid_q.numel() > 0:
                avg_q_val = valid_q.mean().item()
                max_q_val = valid_q.max().item()
                min_q_val = valid_q.min().item()
            else:
                avg_q_val = 0.0
                max_q_val = 0.0
                min_q_val = 0.0

            # 如果 avg/max/min 仍然是非有限，说明已经不是 -inf 的问题，而是网络输出爆炸/溢出
            if not np.isfinite(avg_q_val):
                avg_q_val = 0.0
            if not np.isfinite(max_q_val):
                max_q_val = 0.0
            if not np.isfinite(min_q_val):
                min_q_val = 0.0

            loss_to_log = loss.item() if torch.isfinite(loss) else 0.0

            self.wandb_logger.log_metrics(
                {
                    "loss": loss_to_log,
                    "avg_reward": batch_rewards.mean().item(),
                    "avg_q_value": avg_q_val,
                    "max_q_value": max_q_val,
                    "min_q_value": min_q_val,
                    "temperature": self.temperature,
                    "bad_pred_q_count": pred_bad,
                    "bad_target_q_count": targ_bad,
                    "dropped_td_pairs": dropped,
                    "did_optimizer_step": 1 if did_step else 0,
                },
                step=self.total_steps
            )

        progress = min(1.0, self.total_steps / self.exploration_total_steps)
        self.temperature = self.temperature_start - progress * (self.temperature_start - self.temperature_min)

    def update_target_net(self, soft=False, tau=0.05):
        if soft:
            for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def set_inference_mode(self):
        """切换到推理模式，不进行探索。"""
        self.is_training = False

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
            sys.stderr.write(f"available_commands: {game_state_obj.available_commands}\n")
            sys.stderr.write(f"screen_type: {game_state_obj.screen_type}\n")
            sys.stderr.write(f"are_potions_full: {game_state_obj.are_potions_full()}\n")
            if hasattr(game_state_obj.screen, 'potions'):
                for potion in game_state_obj.screen.potions:
                    sys.stderr.write(f"Potion in shop: {potion.name}\n")
            for potion in game_state_obj.potions:
                sys.stderr.write(f"Potion in inventory: {potion.name}\n")
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

                # 输出药水和mask情况：
                sys.stderr.write(f"当前药水情况：{game_state_obj.screen.potions}\n")
                sys.stderr.write(f"当前选择列表：{game_state_obj.choice_list}\n")
                sys.stderr.write(f"应用后的 choose_option mask：{masks['choose_option']}\n")
                sys.stderr.write(f"应用后的 action_type mask：{masks['action_type']}\n")


            # potion_idxs = self.choose_index_based_name(game_state.choice_list, 'potion')
            potion_idxs = [i for i, choice in enumerate(game_state_obj.choice_list) if choice == 'potion'] if hasattr(game_state_obj, 'choice_list') else []
            sys.stderr.write(f"Detected potion choice indices: {potion_idxs}\n")
            if potion_idxs and game_state_obj.are_potions_full():
                # 可能同时有好几个药水选项
                for potion_idx in potion_idxs:
                    masks['choose_option'][potion_idx] = 0  # 不能选药水了
                # 如果除了药水之外没有别的选项了，就把choose_option全屏蔽
                choose_mask = masks['choose_option']
                # np底层优化过的函数，判断非零元素个数，比sum快 且更准确
                if np.count_nonzero(choose_mask) == 0: 
                    masks['action_type'][DecomposedActionType.CHOOSE.value] = 0

                # 输出药水和mask情况：
                sys.stderr.write(f"当前药水情况：{potion_idxs}\n")
                sys.stderr.write(f"当前选择列表：{game_state_obj.choice_list}\n")
                sys.stderr.write(f"应用后的 choose_option mask：{masks['choose_option']}\n")
                sys.stderr.write(f"应用后的 action_type mask：{masks['action_type']}\n")

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
