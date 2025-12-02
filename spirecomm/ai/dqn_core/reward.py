from spirecomm.ai.dqn_core.action import BaseAction
from spirecomm.spire.game import Game
from spirecomm.ai.dqn_core.action import DecomposedActionType
from spirecomm.ai.absolute_logger import AbsoluteLogger, LogType

class RewardCalculator:
    """
    根据游戏状态的变化计算奖励值。
    这个类的设计旨在将奖励逻辑与主训练循环分离，使其易于调整和维护。
    """

    def __init__(self):
        # --- 战斗相关奖励 ---
        # 对敌人造成伤害的奖励乘数（每1点伤害）
        self.DAMAGE_DEALT_MULTIPLIER = 1.0
        # 自身受到伤害的惩罚乘数（每1点伤害）
        self.DAMAGE_TAKEN_MULTIPLIER = -2.0  # 通常，受到伤害的惩罚应高于造成伤害的奖励
        # 新增：每个怪物死亡的固定奖励
        self.MONSTER_DEATH_REWARD = 10.0
        # 赢得一场普通战斗的奖励
        self.WIN_BATTLE_REWARD = 30.0
        # 输掉一场战斗的惩罚
        self.LOSE_BATTLE_REWARD = -1000.0

        # --- 资源管理奖励 ---
        # 每浪费1点能量结束回合的惩罚
        self.WASTE_ENERGY_PENALTY = -5.0

        # --- 游戏进程奖励 ---
        # Act 1 (1-17层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT1 = 30.0
        # Act 2 (18-34层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT2 = 60.0
        # Act 3 (35-51层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT3 = 120.0
        # Act 4 (52-55层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT4 = 240.0

        # --- BOSS战特殊奖励 ---
        # 战胜第一幕BOSS (17层) 的额外奖励
        self.WIN_ACT1_BOSS_BONUS = 120.0
        # 战胜第二幕BOSS (34层) 的额外奖励
        self.WIN_ACT2_BOSS_BONUS = 240.0
        # 战胜第三幕BOSS (51层) 的额外奖励
        self.WIN_ACT3_BOSS_BONUS = 480.0
        # 战胜最终BOSS (55层心脏) 的巨大奖励
        self.WIN_FINAL_BOSS_REWARD = 10000.0

        

        self.absolute_logger = AbsoluteLogger(LogType.REWARD)
        self.absolute_logger.start_episode()

    def calculate(self, prev_state: Game, next_state:Game, action:BaseAction=None):
        """
        计算从 prev_state 转换到 next_state 所获得的奖励，并将各项明细通过 absolute_logger.write 输出，方便排查。
        """
        total_reward = 0.0

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
                contributions.append(("damage_dealt", value, f"damage={damage_dealt} * mul={self.DAMAGE_DEALT_MULTIPLIER}"))

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
                contributions.append(("monster_deaths", value, f"count={dead_count} * per={self.MONSTER_DEATH_REWARD}"))

            # 计算自身受到的伤害
            if prev_state.player is not None and next_state.player is not None:
                damage_taken = prev_state.player.current_hp - next_state.player.current_hp
                if damage_taken > 0:
                    value = damage_taken * self.DAMAGE_TAKEN_MULTIPLIER
                    total_reward += value
                    contributions.append(("damage_taken", value, f"damage={damage_taken} * mul={self.DAMAGE_TAKEN_MULTIPLIER}"))

            # 检查是否浪费能量 (当执行 "END" 动作时触发)
            if action is not None and getattr(action, "decomposed_type", None) == DecomposedActionType.END and prev_state.player is not None:
                wasted_energy = prev_state.player.energy
                if wasted_energy > 0:
                    value = wasted_energy * self.WASTE_ENERGY_PENALTY
                    total_reward += value
                    contributions.append(("wasted_energy", value, f"wasted={wasted_energy} * mul={self.WASTE_ENERGY_PENALTY}"))

        # --- 2. 战斗结果奖励 ---
        # 赢得战斗: 从战斗状态进入非战斗状态
        if prev_state.in_combat and not next_state.in_combat:
            # 检查是否是最终BOSS战胜利
            if prev_state.floor == 55:
                value = self.WIN_FINAL_BOSS_REWARD
                total_reward += value
                contributions.append(("win_final_boss", value, f"floor={prev_state.floor}"))
            else:
                value = self.WIN_BATTLE_REWARD
                total_reward += value
                contributions.append(("win_battle", value, f"floor={prev_state.floor}"))
                # 检查是否是幕BOSS战胜利
                if prev_state.floor == 17:
                    bv = self.WIN_ACT1_BOSS_BONUS
                    total_reward += bv
                    contributions.append(("win_act1_boss_bonus", bv, "floor=17"))
                elif prev_state.floor == 34:
                    bv = self.WIN_ACT2_BOSS_BONUS
                    total_reward += bv
                    contributions.append(("win_act2_boss_bonus", bv, "floor=34"))
                elif prev_state.floor == 51:
                    bv = self.WIN_ACT3_BOSS_BONUS
                    total_reward += bv
                    contributions.append(("win_act3_boss_bonus", bv, "floor=51"))

        # 输掉战斗: 游戏结束
        if prev_state.in_game and not next_state.in_game:
            value = self.LOSE_BATTLE_REWARD
            total_reward += value
            contributions.append(("lose_game", value, "从 in_game 到 not in_game"))

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
            contributions.append(("floor_increase", value, f"change={floor_change} * per={per} (band_based_on_floor={target_floor_for_band})"))

        # 输出每一项贡献及总和，便于定位问题
        log_lines = ["奖励明细："]
        for name, val, detail in contributions:
            log_lines.append(f"  - {name}: {val} ({detail})")
        log_lines.append(f"总奖励: {total_reward}\n\n")
        self.absolute_logger.write("\n".join(log_lines))

        return total_reward