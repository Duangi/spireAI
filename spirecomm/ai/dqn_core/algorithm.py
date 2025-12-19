from calendar import c
from dataclasses import fields
import random
from collections import deque
from re import purge
import sys
import time
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
        # 训练速度统计：60s 窗口 + 1h 窗口
        self._velocity_last_time = None
        self._velocity_last_step = 0
        self._velocity_last_flush_time = None
        self._velocity_last_flush_step = 0
        self._hour_last_time = None
        self._hour_last_step = 0
        # 训练速度统计：记录最近一次统计的 wall-clock 时间与 step
        self.last_train_time = None
        self.last_episodes = None
        
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

        self.temperature_start = 1.44
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

            # 训练速度：按固定窗口统计（更平滑）
            now_time = datetime.datetime.now()
            steps_per_min = 0.0
            steps_per_hour = 0.0

            try:
                # 初始化
                if self._velocity_last_time is None:
                    self._velocity_last_time = now_time
                    self._velocity_last_step = self.total_steps
                    self._velocity_last_flush_time = now_time
                    self._velocity_last_flush_step = self.total_steps
                    self._hour_last_time = now_time
                    self._hour_last_step = self.total_steps
                else:
                    # ---- 60 秒窗口：每满 60 秒更新一次 steps/min ----
                    if self._velocity_last_flush_time is not None:
                        flush_diff = (now_time - self._velocity_last_flush_time).total_seconds()
                        if flush_diff >= 60.0:
                            step_diff = self.total_steps - self._velocity_last_flush_step
                            if flush_diff > 0 and step_diff >= 0:
                                steps_per_min = (step_diff / flush_diff) * 60.0

                            # 滚动 60s 窗口起点
                            self._velocity_last_flush_time = now_time
                            self._velocity_last_flush_step = self.total_steps

                    # ---- 1 小时窗口：每满 3600 秒更新一次 steps/hour（过去一小时总 step 数） ----
                    if self._hour_last_time is not None:
                        hour_diff = (now_time - self._hour_last_time).total_seconds()
                        if hour_diff >= 3600.0:
                            hour_step_diff = self.total_steps - self._hour_last_step
                            if hour_step_diff >= 0:
                                steps_per_hour = float(hour_step_diff)

                            # 滚动 1h 窗口起点
                            self._hour_last_time = now_time
                            self._hour_last_step = self.total_steps
            except Exception:
                pass

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
                    "train/steps_per_min": float(steps_per_min),
                    "train/steps_per_hour": float(steps_per_hour),
                },
                step=self.total_steps
            )
            now_time = datetime.datetime.now()
            self.last_episodes = self.total_steps
            self.last_train_time = now_time

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

        # 先记录 available_commands（你要求在动作选择时输出）
        ava_commands = getattr(game_state_obj, 'available_commands', None)
        self.absolute_logger.write(f"Available Commands: {ava_commands}\n")

        # 如果在战斗中：输出能量、怪物血量、意图/攻击
        try:
            in_combat = bool(getattr(game_state_obj, 'in_combat', False))
            room_phase = getattr(game_state_obj, 'room_phase', None)
            if room_phase is not None:
                try:
                    # RoomPhase.COMBAT
                    in_combat = in_combat or (getattr(room_phase, 'name', None) == 'COMBAT')
                except Exception:
                    pass

            if in_combat:
                # 能量：优先用 player.energy（在 Game 里更稳定）
                energy = None
                if hasattr(game_state_obj, 'player') and getattr(game_state_obj.player, 'energy', None) is not None:
                    energy = game_state_obj.player.energy
                else:
                    for attr in ('energy', 'current_energy', 'player_energy'):
                        if hasattr(game_state_obj, attr):
                            energy = getattr(game_state_obj, attr)
                            break

                self.absolute_logger.write(f"[Combat] Energy: {energy}\n")

                # 怪物信息
                monsters = getattr(game_state_obj, 'monsters', []) or []
                self.absolute_logger.write(f"[Combat] Monsters: {len(monsters)}\n")

                for mi, m in enumerate(monsters):
                    try:
                        m_name = getattr(m, 'name', f"Monster_{mi}")
                        m_hp = getattr(m, 'current_hp', None)
                        m_max = getattr(m, 'max_hp', None)

                        # intent：Monster.intent 是 Intent Enum
                        intent = getattr(m, 'intent', None)
                        intent_name = None
                        try:
                            intent_name = intent.name if intent is not None and hasattr(intent, 'name') else str(intent)
                        except Exception:
                            intent_name = str(intent)

                        # 攻击信息：使用 Monster.move_adjusted_damage / move_base_damage / move_hits
                        dmg = None
                        hits = None

                        if getattr(m, 'move_adjusted_damage', None) is not None:
                            dmg = m.move_adjusted_damage
                        elif getattr(m, 'move_base_damage', None) is not None:
                            dmg = m.move_base_damage

                        hits = getattr(m, 'move_hits', None)

                        # intent 是攻击类时，优先输出攻击；否则也把 intent 打出来
                        is_attack_intent = False
                        try:
                            is_attack_intent = bool(intent is not None and hasattr(intent, 'is_attack') and intent.is_attack())
                        except Exception:
                            # 有的 intent 可能没有 is_attack
                            is_attack_intent = False

                        if is_attack_intent and dmg is not None:
                            if hits is not None and int(hits) > 1:
                                self.absolute_logger.write(f"[Combat]  - {mi}:{m_name} HP={m_hp}/{m_max} Intent={intent_name} ATK={int(dmg)}x{int(hits)}\n")
                            else:
                                self.absolute_logger.write(f"[Combat]  - {mi}:{m_name} HP={m_hp}/{m_max} Intent={intent_name} ATK={int(dmg)}\n")
                        else:
                            # 非攻击或未知攻击值：只输出意图
                            self.absolute_logger.write(f"[Combat]  - {mi}:{m_name} HP={m_hp}/{m_max} Intent={intent_name}\n")

                    except Exception as e:
                        self.absolute_logger.write(f"[Combat]  - monster[{mi}] parse failed: {e}\n")
        except Exception:
            # 如果拿不到字段，也别影响选动作
            pass

        # 保存一份未 mask 的 q_type 用于调试显示（不影响后续逻辑）
        raw_q_type = q_type.detach().clone()

        # 应用 action_type mask
        q_type[~action_type_mask] = -float('inf')

        # --- 动作类型选择 ---
        probs = None
        if self.is_training:
            probs = torch.softmax(q_type / self.temperature, dim=-1)
            # 二次检查：防止 probs 含有 NaN (例如 q_type 全是 -inf)
            if torch.isnan(probs).any() or probs.sum() == 0:
                # 均匀分布回退
                probs = torch.zeros_like(probs)
                probs[action_type_mask] = 1.0 / action_type_mask.sum().float()
            
            act_idx = torch.multinomial(probs, 1).item()
        else:
            # 推理模式下也算一份 probs 方便日志阅读
            probs = torch.softmax(q_type / max(self.temperature, 1e-6), dim=-1)
            if torch.isnan(probs).any() or probs.sum() == 0:
                probs = torch.zeros_like(probs)
                probs[action_type_mask] = 1.0 / action_type_mask.sum().float()
            act_idx = torch.argmax(q_type).item()

        action_type = DecomposedActionType(act_idx)

        # --- 输出：mask 后每个动作的 Q 值与概率 ---
        try:
            q_type_cpu = q_type.detach().float().cpu().numpy().tolist()
            probs_cpu = probs.detach().float().cpu().numpy().tolist() if probs is not None else None
            mask_cpu = action_type_mask.detach().cpu().numpy().astype(bool).tolist()

            self.absolute_logger.write("[ActionType] After mask: Q / Prob (mask=1 only)\n")
            for i in range(len(q_type_cpu)):
                m = bool(mask_cpu[i]) if i < len(mask_cpu) else False
                if not m:
                    continue

                try:
                    name = DecomposedActionType(i).name
                except Exception:
                    name = f"Type_{i}"

                qv = q_type_cpu[i]
                pv = probs_cpu[i] if (probs_cpu is not None and i < len(probs_cpu)) else 0.0
                self.absolute_logger.write(f"  - {name:<16} masked_q={qv:.6f} prob={pv:.6f}\n")
        except Exception as e:
            self.absolute_logger.write(f"[ActionType] Log failed: {e}\n")

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

            # 计算概率（用于输出）
            p = torch.softmax(q_vals / max(self.temperature, 1e-6), dim=-1)
            if torch.isnan(p).any() or p.sum() == 0:
                p = torch.zeros_like(p)
                if mask.any():
                    p[mask] = 1.0 / mask.sum().float()

            # 输出该分支的 Q 与概率（只输出 mask=1，并额外输出名称）
            try:
                q_cpu = q_vals.detach().float().cpu().numpy().tolist()
                p_cpu = p.detach().float().cpu().numpy().tolist()
                m_cpu = mask.detach().cpu().numpy().astype(bool).tolist()

                # 名称解析
                def _name_for_idx(idx: int) -> str:
                    try:
                        if mask_key == 'play_card':
                            if hasattr(game_state_obj, 'hand') and idx < len(game_state_obj.hand):
                                return str(getattr(game_state_obj.hand[idx], 'name', f"Card_{idx}"))
                            return f"Card_{idx}"
                        if mask_key == 'target_monster':
                            if hasattr(game_state_obj, 'monsters') and idx < len(game_state_obj.monsters):
                                return str(getattr(game_state_obj.monsters[idx], 'name', f"Monster_{idx}"))
                            return f"Monster_{idx}"
                        if mask_key == 'choose_option':
                            if hasattr(game_state_obj, 'choice_list') and idx < len(game_state_obj.choice_list):
                                return str(game_state_obj.choice_list[idx])
                            return f"Choice_{idx}"
                        if mask_key == 'potion_use' or mask_key == 'potion_discard':
                            if hasattr(game_state_obj, 'potions') and idx < len(game_state_obj.potions):
                                return str(getattr(game_state_obj.potions[idx], 'name', f"Potion_{idx}"))
                            return f"Potion_{idx}"
                    except Exception:
                        pass
                    return f"idx_{idx}"

                self.absolute_logger.write(f"[{mask_key}] After mask: Q / Prob (mask=1 only)\n")
                for i in range(len(q_cpu)):
                    m = bool(m_cpu[i]) if i < len(m_cpu) else False
                    if not m:
                        continue
                    qv = q_cpu[i]
                    pv = p_cpu[i] if i < len(p_cpu) else 0.0
                    nm = _name_for_idx(i)
                    self.absolute_logger.write(f"  - idx={i:<2} name={nm} masked_q={qv:.6f} prob={pv:.6f}\n")
            except Exception as e:
                self.absolute_logger.write(f"[{mask_key}] Log failed: {e}\n")

            if self.is_training:
                return torch.multinomial(p, 1).item()
            else:
                return torch.argmax(q_vals).item()

        # 
        # 根据类型组装 Action 对象
        if action_type == DecomposedActionType.PLAY:
            hand_idx = select_idx(q_card, 'play_card')
            
            target_idx = None
            # 判断是否需要目标
            card = game_state_obj.hand[hand_idx]
            if card.has_target:
                target_idx = select_idx(q_monster, 'target_monster')

            act = PlayAction(ActionType.PLAY, hand_idx, target_idx)
            # 输出最终动作
            try:
                card_name = game_state_obj.hand[hand_idx].name if hand_idx < len(game_state_obj.hand) else str(hand_idx)
                tgt_name = None
                if target_idx is not None:
                    tgt_name = game_state_obj.monsters[target_idx].name if target_idx < len(game_state_obj.monsters) else str(target_idx)
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} card_idx={hand_idx} card={card_name} target_idx={target_idx} target={tgt_name}\n\n")
            except Exception:
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} card_idx={hand_idx} target_idx={target_idx}\n\n")
            return act
            
        elif action_type == DecomposedActionType.CHOOSE:
            # 检查是否是 Shop 进入逻辑
            if game_state_obj.screen_type == ScreenType.SHOP_ROOM:
                # 站在商店门口
                # ... (简化逻辑，直接选)
                pass
            
            choice_idx = select_idx(q_choice, 'choose_option')
            act = ChooseAction(ActionType.CHOOSE, choice_idx)
            try:
                choice_name = None
                if hasattr(game_state_obj, 'choice_list') and choice_idx < len(game_state_obj.choice_list):
                    choice_name = str(game_state_obj.choice_list[choice_idx])
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} choice_idx={choice_idx} choice={choice_name}\n\n")
            except Exception:
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} choice_idx={choice_idx}\n\n")
            return act
            
        elif action_type == DecomposedActionType.POTION_USE:
            pot_idx = select_idx(q_pot_use, 'potion_use')
            target_idx = None
            if game_state_obj.potions[pot_idx].requires_target:
                target_idx = select_idx(q_monster, 'target_monster')
            act = PotionUseAction(ActionType.POTION_USE, pot_idx, target_idx)
            
            try:
                pot_name = game_state_obj.potions[pot_idx].name if pot_idx < len(game_state_obj.potions) else str(pot_idx)
                tgt_name = None
                if target_idx is not None:
                    tgt_name = game_state_obj.monsters[target_idx].name if target_idx < len(game_state_obj.monsters) else str(target_idx)
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} pot_idx={pot_idx} pot={pot_name} target_idx={target_idx} target={tgt_name}\n\n")
            except Exception:
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} pot_idx={pot_idx} target_idx={target_idx}\n\n")
            return act
            
        elif action_type == DecomposedActionType.POTION_DISCARD:
            pot_idx = select_idx(q_pot_disc, 'potion_discard')
            act = PotionDiscardAction(ActionType.POTION_DISCARD, pot_idx)
            try:
                pot_name = game_state_obj.potions[pot_idx].name if pot_idx < len(game_state_obj.potions) else str(pot_idx)
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} pot_idx={pot_idx} pot={pot_name}\n\n")
            except Exception:
                self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name} pot_idx={pot_idx}\n\n")
            return act
            
        else:
            # End, Proceed, etc.
            base_type = action_type.to_action_type()
            from spirecomm.ai.dqn_core.action import SingleAction
            act = SingleAction(base_type, decomposed_type=action_type)
            self.absolute_logger.write(f"[SelectedAction] {act.decomposed_type.name}\n\n")
            return act
