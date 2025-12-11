from typing import List
from spirecomm.ai.dqn_core.action import BaseAction
from spirecomm.spire.game import Game
from spirecomm.ai.dqn_core.action import DecomposedActionType
from spirecomm.ai.absolute_logger import AbsoluteLogger, LogType
# 用于在需要时生成 game.state_hash
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.spire.screen import CombatReward, CombatRewardScreen, ScreenType

class RewardCalculator:
    """
    根据游戏状态的变化计算奖励值。
    这个类的设计旨在将奖励逻辑与主训练循环分离，使其易于调整和维护。
    """

    def __init__(self, state_processor=None):
        self.state_processor = state_processor
        # --- 战斗相关奖励 ---
        # 对敌人造成伤害的奖励乘数（每1点伤害）
        self.DAMAGE_DEALT_MULTIPLIER = 1.0
        # 自身受到伤害的惩罚乘数（每1点伤害）
        self.DAMAGE_TAKEN_MULTIPLIER = -2.0 
        # 新增：每个怪物死亡的固定奖励
        self.MONSTER_DEATH_REWARD = 5.0
        # 赢得一场普通战斗的奖励
        self.WIN_BATTLE_REWARD = 50.0
        # 输掉一场战斗的惩罚 -1000还是太高了，哥们直接不打了 原地摆烂
        self.LOSE_BATTLE_REWARD = -10.0

        # --- 资源管理奖励 ---
        # 每浪费1点能量结束回合的惩罚
        self.WASTE_ENERGY_PENALTY = -1.0 # 降低惩罚，初期随机策略容易浪费能量
        # 每获得1点金钱的奖励  0.1的时候战斗结束的钱都不捡了
        self.GOLD_GAINED_REWARD = 0.5
        # 每被偷1点钱的惩罚
        self.GOLD_STOLEN_PENALTY = -0.5
        # 每获得一瓶药水的奖励
        self.POTION_GAINED_REWARD = 5.0
        # 每失去一瓶药水的惩罚
        self.POTION_DISCARD_PENALTY = -5.0

        # 给一个赢得了战斗但是不捡金币的惩罚！浪费可耻
        self.WIN_BATTLE_NO_GOLD_PENALTY = -1.0
        # 赢得了战斗不捡遗物的惩罚
        self.WIN_BATTLE_NO_RELIC_PENALTY = -5.0
        # 赢得了战斗不捡药水的惩罚
        self.WIN_BATTLE_NO_POTION_PENALTY = -3.0

        # --- 游戏进程奖励 ---
        # Act 1 (1-17层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT1 = 10.0
        # Act 2 (18-34层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT2 = 20.0
        # Act 3 (35-51层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT3 = 30.0
        # Act 4 (52-55层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT4 = 40.0

        # --- BOSS战特殊奖励 ---
        # 战胜第一幕BOSS (17层) 的额外奖励
        self.WIN_ACT1_BOSS_BONUS = 1000.0
        # 战胜第二幕BOSS (34层) 的额外奖励
        self.WIN_ACT2_BOSS_BONUS = 3000.0
        # 战胜第三幕BOSS (51层) 的额外奖励
        self.WIN_ACT3_BOSS_BONUS = 6000.0
        # 战胜最终BOSS (55层心脏) 的巨大奖励
        self.WIN_FINAL_BOSS_REWARD = 10000.0

        # --- 卡bug专用奖惩 ---
        # 战斗时应该果断play/end,而不是反复choose动作
        self.CHOOSE_PENALTY_IN_COMBAT = -2.0
        # 战斗中选择了CONFIRM时给予奖励
        self.CONFIRM_REWARD_IN_COMBAT = 2.0
        # 假设next_state和prev_prev_state完全一致的话，表示卡bug不动了，给予大大的惩罚！
        self.STUCK_PENALTY = -50.0
        self.stuck_count = 5


        self.absolute_logger = AbsoluteLogger(LogType.REWARD)
        self.absolute_logger.start_episode()

    def calculate(self, prev_state: Game, next_state:Game, action:BaseAction=None, prev_prev_state: Game=None):
        """
        计算从 prev_state 转换到 next_state 所获得的奖励，并将各项明细通过 absolute_logger.write 输出，方便排查。
        """
        total_reward = 0.0

        # 确保传入的 state 有 state_hash（best-effort：如果没有则生成）
        def _ensure_hash(g):
            if g is None:
                self.absolute_logger.write("无法为 None 状态生成 hash，直接返回 0.0")
                return
            if getattr(g, "state_hash", None) is None:
                try:
                    # 使用传入的 processor 或短生命周期的 processor 生成并写回 g.state_hash
                    proc = self.state_processor if self.state_processor else GameStateProcessor()
                    proc.get_state_tensor(g)
                except Exception as e:
                    # 容错：记录但不抛出，避免影响训练流程
                    try:
                        self.absolute_logger.write(f"无法计算 state_hash: {e}")
                    except Exception:
                        pass

        # 先为 prev_prev/prev/next 尝试生成 hash（若缺失）
        _ensure_hash(prev_prev_state)
        _ensure_hash(prev_state)
        _ensure_hash(next_state)

        # 确保状态有效
        if prev_state is None or next_state is None:
            self.absolute_logger.write("Reward 计算：prev_state 或 next_state 为 None，返回 0.0")
            return 0.0

        # 先输出前后状态的摘要，方便查看关键字段
        try:
            prev_monsters_hp = sum(m.current_hp for m in prev_state.monsters) if getattr(prev_state, "monsters", None) else 0
            next_monsters_hp = sum(m.current_hp for m in next_state.monsters) if getattr(next_state, "monsters", None) else 0
            prev_player_hp = prev_state.player.current_hp if getattr(prev_state, "player", None) else "N/A"
            next_player_hp = next_state.player.current_hp if getattr(next_state, "player", None) else "N/A"
            prev_player_energy = prev_state.player.energy if getattr(prev_state, "player", None) else "N/A"
            next_player_energy = next_state.player.energy if getattr(next_state, "player", None) else "N/A"
        except Exception as e:
            # 防护：尽量不要因为日志导致异常
            self.absolute_logger.write(f"状态摘要提取异常: {e}")
            prev_monsters_hp = next_monsters_hp = 0
            prev_player_hp = next_player_hp = prev_player_energy = next_player_energy = "N/A"

        # 捕获并格式化当前动作，直接使用 action.to_string()，回退到简单描述以防异常
        try:
            if action is None:
                action_str = "None"
            else:
                # 直接使用 action 提供的 to_string 方法（统一输出格式，如 "play 4 0"）
                if hasattr(action, "to_string"):
                    action_str = action.to_string()
                else:
                    action_str = repr(action)
        except Exception:
            action_str = "action_format_error"

        # 将动作信息加入状态摘要
        state_summary = (
            f"状态摘要：\n"
            f"  前: floor={prev_state.floor} in_game={prev_state.in_game} in_combat={prev_state.in_combat} "
            f"player_hp={prev_player_hp} player_energy={prev_player_energy} monsters_hp_total={prev_monsters_hp}\n"
            f"  后: floor={next_state.floor} in_game={next_state.in_game} in_combat={next_state.in_combat} "
            f"player_hp={next_player_hp} player_energy={next_player_energy} monsters_hp_total={next_monsters_hp}\n"
            f"  动作: {action_str}\n"
        )
        self.absolute_logger.write(state_summary)

        # 用于保存各项贡献，便于逐项输出
        contributions = []

        # --- 1. 战斗内奖励 (尽量在 prev_state.in_combat 时计算，避免 in_game 标识不一致导致跳过) ---
        # 如果 in_combat 为 True 但 in_game 标识不满足同时为 True，记录提示但继续计算
        if prev_state.in_combat and not (prev_state.in_game and next_state.in_game):
            self.absolute_logger.write(
                "提示: 前状态 in_combat=True 但 in_game 标识不一致（可能为 False）；仍尝试计算战斗内伤害/受伤/wasted_energy/死亡奖励。"
                f" flags: prev.in_game={prev_state.in_game}, next.in_game={next_state.in_game}"
            )

        # 只要 prev_state.in_combat 为 True，就计算战斗内变化（覆盖之前严格要求的 in_game 条件）
        if prev_state.in_combat:
            # 计算对敌人造成的总伤害
            prev_total_hp = prev_monsters_hp
            next_total_hp = next_monsters_hp
            damage_dealt = prev_total_hp - next_total_hp
            if damage_dealt > 0:
                value = damage_dealt * self.DAMAGE_DEALT_MULTIPLIER
                total_reward += value
                contributions.append(("造成伤害", value, f"damage={damage_dealt} * mul={self.DAMAGE_DEALT_MULTIPLIER}"))

            # 计算怪物死亡（is_gone 或 被移除）
            prev_monsters = getattr(prev_state, "monsters", []) or []
            next_monsters = getattr(next_state, "monsters", []) or []
            dead_count = 0
            for i, pm in enumerate(prev_monsters):
                prev_gone = getattr(pm, "is_gone", False)
                # 如果 next 中存在对应索引，检测 is_gone 标记变化或 hp 从 >0 变为 0
                if i < len(next_monsters):
                    nm = next_monsters[i]
                    next_gone = getattr(nm, "is_gone", False)
                    next_hp = getattr(nm, "current_hp", None)
                    prev_hp = getattr(pm, "current_hp", None)
                    if (not prev_gone) and next_gone:
                        dead_count += 1
                    # 作为补充，如果 is_gone 不可靠，也考虑 hp 变为 0 或 None
                    elif (not prev_gone) and (next_hp == 0 and prev_hp is not None and prev_hp > 0):
                        dead_count += 1
                else:
                    # 如果 next_state 中没有该索引，视为该怪物已被移除（死亡）
                    if not prev_gone:
                        dead_count += 1

            if dead_count > 0:
                value = dead_count * self.MONSTER_DEATH_REWARD
                total_reward += value
                contributions.append(("怪物死亡", value, f"count={dead_count} * per={self.MONSTER_DEATH_REWARD}"))

            # 计算自身受到的伤害
            if prev_state.player is not None and next_state.player is not None:
                damage_taken = prev_state.player.current_hp - next_state.player.current_hp
                if damage_taken > 0:
                    value = damage_taken * self.DAMAGE_TAKEN_MULTIPLIER
                    total_reward += value
                    contributions.append(("受到伤害", value, f"damage={damage_taken} * mul={self.DAMAGE_TAKEN_MULTIPLIER}"))

            # 检查是否浪费能量 (当执行 "END" 动作时触发)
            if action is not None and getattr(action, "decomposed_type", None) == DecomposedActionType.END and prev_state.player is not None:
                wasted_energy = prev_state.player.energy
                if wasted_energy > 0:
                    value = wasted_energy * self.WASTE_ENERGY_PENALTY
                    total_reward += value
                    contributions.append(("浪费能量", value, f"wasted={wasted_energy} * mul={self.WASTE_ENERGY_PENALTY}"))

        # --- 2. 战斗结果奖励 ---
        # 赢得战斗: 从战斗状态进入非战斗状态
        if prev_state.in_combat and not next_state.in_combat:
            # 检查是否是最终BOSS战胜利
            if prev_state.floor == 55:
                value = self.WIN_FINAL_BOSS_REWARD
                total_reward += value
                contributions.append(("赢得最终BOSS", value, f"floor={prev_state.floor}"))
            else:
                value = self.WIN_BATTLE_REWARD
                total_reward += value
                contributions.append(("赢得战斗", value, f"floor={prev_state.floor}"))
                # 检查是否是幕BOSS战胜利
                if prev_state.floor == 17:
                    bv = self.WIN_ACT1_BOSS_BONUS
                    total_reward += bv
                    contributions.append(("赢得第一幕BOSS奖励", bv, "floor=17"))
                elif prev_state.floor == 34:
                    bv = self.WIN_ACT2_BOSS_BONUS
                    total_reward += bv
                    contributions.append(("赢得第二幕BOSS奖励", bv, "floor=34"))
                elif prev_state.floor == 51:
                    bv = self.WIN_ACT3_BOSS_BONUS
                    total_reward += bv
                    contributions.append(("赢得第三幕BOSS奖励", bv, "floor=51"))
        # 输掉战斗: 游戏结束
        if prev_state.in_game and not next_state.in_game:
            value = self.LOSE_BATTLE_REWARD
            total_reward += value
            contributions.append(("输掉游戏", value, "从 in_game 到 not in_game"))

        # --- 3. 游戏进程奖励 ---
        # 使用 next_state.floor（到达后的楼层）来决定属于哪个 Act 的奖励区间，避免 prev_state.floor 为 0 时误判为 Act4
        floor_change = next_state.floor - prev_state.floor
        if floor_change > 0:
            # 以到达后的楼层为判定依据（若 next_state.floor 为 0 则退回使用 prev_state.floor）
            target_floor_for_band = next_state.floor if next_state.floor > 0 else prev_state.floor
            if 1 <= target_floor_for_band <= 17:
                per = self.FLOOR_INCREASE_ACT1
            elif 18 <= target_floor_for_band <= 34:
                per = self.FLOOR_INCREASE_ACT2
            elif 35 <= target_floor_for_band <= 51:
                per = self.FLOOR_INCREASE_ACT3
            else: # 52层及以上
                per = self.FLOOR_INCREASE_ACT4
            value = floor_change * per
            total_reward += value
            contributions.append(("层数上升", value, f"change={floor_change} * per={per} (band_based_on_floor={target_floor_for_band})"))

        # --- 4. 资源管理奖励 ---
        # 金钱变化奖励
        gold_change = next_state.gold - prev_state.gold
        if gold_change > 0:
            value = gold_change * self.GOLD_GAINED_REWARD
            total_reward += value
            contributions.append(("获得金币", value, f"change={gold_change} * mul={self.GOLD_GAINED_REWARD}"))
        elif gold_change < 0:
            # 战斗中掉钱惩罚（大概率是被小偷偷了）
            if prev_state.in_combat and next_state.in_combat:
                value = abs(gold_change) * self.GOLD_STOLEN_PENALTY
                total_reward += value
                contributions.append(("战斗中失去金币惩罚", value, f"change={gold_change} * mul={self.GOLD_STOLEN_PENALTY}"))
        # 药水变化奖励
        # 需要判断potions数组中，id不为"Potion Slot" 或者 name 不为 "药水栏"的数量变
        prev_potion_count = 0
        for potion in prev_state.potions:
            if potion.potion_id != "Potion Slot" or potion.name != "药水栏":
                prev_potion_count += 1
        next_potion_count = 0
        for potion in next_state.potions:
            if potion.potion_id != "Potion Slot" or potion.name != "药水栏":
                next_potion_count += 1
        potion_change = next_potion_count - prev_potion_count
        if potion_change > 0:
            value = potion_change * self.POTION_GAINED_REWARD
            total_reward += value
            contributions.append(("获得药水", value, f"change={potion_change} * mul={self.POTION_GAINED_REWARD}"))
        elif potion_change < 0:
            if action.decomposed_type == DecomposedActionType.POTION_DISCARD:
                # 扔药水惩罚,且当前screen不是战斗奖励选择界面和商店界面
                if prev_state.screen_type != ScreenType.COMBAT_REWARD and prev_state.screen_type != ScreenType.SHOP_SCREEN:
                    value = abs(potion_change) * self.POTION_DISCARD_PENALTY
                    total_reward += value
                    contributions.append(("除了战斗奖励界面和商店界面丢弃药水 给予惩罚", value, f"{self.POTION_DISCARD_PENALTY}"))
            elif action.decomposed_type == DecomposedActionType.POTION_USE:
                # 使用药水不惩罚
                pass
        
        # 赢得战斗但没有捡金币的惩罚
        if prev_state.screen_type == "CombatRewardScreen":
            if "gold" in prev_state.choice_list:
                # 走了但是选项里面还有金币
                value = self.WIN_BATTLE_NO_GOLD_PENALTY
                total_reward += value
                contributions.append(("战斗后未捡金币惩罚", value, f"{self.WIN_BATTLE_NO_GOLD_PENALTY}"))
            if "relic" in prev_state.choice_list:
                # 走了但是选项里面还有遗物
                value = self.WIN_BATTLE_NO_RELIC_PENALTY
                total_reward += value
                contributions.append(("战斗后未捡遗物惩罚", value, f"{self.WIN_BATTLE_NO_RELIC_PENALTY}"))
            if "potion" in prev_state.choice_list and not prev_state.are_potions_full():
                # 走了但是选项里面还有药水，视为没捡
                value = self.WIN_BATTLE_NO_POTION_PENALTY
                total_reward += value
                contributions.append(("战斗后药水栏没满但是没捡药水惩罚", value, f"{self.WIN_BATTLE_NO_POTION_PENALTY}"))
        # --- 6. 战斗中选择动作的奖惩 ---
        
        # if prev_state.in_combat and next_state.in_combat and action is not None:
        #     # 前后都是choose动作，给予惩罚
        #     if action.decomposed_type == DecomposedActionType.CHOOSE:
        #         value = self.CHOOSE_PENALTY_IN_COMBAT
        #         total_reward += value
        #         contributions.append(("战斗中选择动作惩罚", value, f"固定值={self.CHOOSE_PENALTY_IN_COMBAT}"))
        #     # 战斗中选择了confirm动作，给予奖励
        #     elif action.decomposed_type == DecomposedActionType.CONFIRM:
        #         value = self.CONFIRM_REWARD_IN_COMBAT
        #         total_reward += value
        #         contributions.append(("战斗中确认动作奖励", value, f"固定值={self.CONFIRM_REWARD_IN_COMBAT}"))
        # --- 7. 卡bug检测 ---
        # 因为需要打开选择界面来查看卡牌或者之类奖励的情况，由于卡牌不太好需要skip，此时也会导致prev_prev_state和next_state相同
        # 因此这里需要给一个卡bug的次数阈值，暂定5次，因为有一个同时选5张牌的遗物。
        # 这个次数阈值在层数变化时重置
        is_stuck = False
        contributions.append(("prev_state.floor", prev_state.floor, "用于检测层数变化以重置计数器"))
        contributions.append(("next_state.floor", next_state.floor, "用于检测层数变化以重置计数器"))
        if next_state.floor != prev_state.floor:
            self.stuck_count = 5
        if prev_prev_state is not None:
            # 简单比较 prev_prev_state 和 next_state 的关键字段是否完全一致
            try:
                contributions.append(("prev_prev_state_hash", prev_prev_state.state_hash, "prev_prev_state 的 hash 值"))
                contributions.append(("next_state_hash", next_state.state_hash, "next_state 的 hash 值"))
                # 使用 hash 值进行比较，而不是对象比较
                if prev_prev_state.state_hash == next_state.state_hash:
                    contributions.append(("卡bug检测 count", self.stuck_count, "prev_prev_state 与 next_state 关键字段完全一致，开始计数"))
                    self.stuck_count -= 1
                if self.stuck_count < 0:
                    is_stuck = True
            except Exception:
                is_stuck = False  # 如果比较过程中出错，视为未卡住

            if is_stuck:
                value = self.STUCK_PENALTY
                total_reward += value
                contributions.append(("卡bug惩罚", value, "prev_prev_state 与 next_state 关键字段完全一致"))
        # 输出每一项贡献及总和，便于定位问题
        log_lines = ["\n奖励明细："]
        details_list = []
        for name, val, detail in contributions:
            log_lines.append(f"  - {name}: {val} ({detail})")
            details_list.append(f"{name}: {val}")
        log_lines.append(f"总奖励: {total_reward}\n\n")
        self.absolute_logger.write("\n".join(log_lines))

        # Return both total reward and details string
        details_str = ", ".join(details_list)
        return total_reward, details_str