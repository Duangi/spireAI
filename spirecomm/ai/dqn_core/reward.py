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
        # 赢得一场普通战斗的奖励
        self.WIN_BATTLE_REWARD = 100.0
        # 输掉一场战斗的惩罚
        self.LOSE_BATTLE_REWARD = -500.0

        # --- 资源管理奖励 ---
        # 每浪费1点能量结束回合的惩罚
        self.WASTE_ENERGY_PENALTY = -5.0

        # --- 游戏进程奖励 ---
        # Act 1 (1-17层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT1 = 5.0
        # Act 2 (18-34层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT2 = 10.0
        # Act 3 (35-51层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT3 = 15.0
        # Act 4 (52-55层) 每提升一层的奖励
        self.FLOOR_INCREASE_ACT4 = 30.0

        # --- BOSS战特殊奖励 ---
        # 战胜第一幕BOSS (17层) 的额外奖励
        self.WIN_ACT1_BOSS_BONUS = 200.0
        # 战胜第二幕BOSS (34层) 的额外奖励
        self.WIN_ACT2_BOSS_BONUS = 400.0
        # 战胜第三幕BOSS (51层) 的额外奖励
        self.WIN_ACT3_BOSS_BONUS = 600.0
        # 战胜最终BOSS (55层心脏) 的巨大奖励
        self.WIN_FINAL_BOSS_REWARD = 2000.0

    def calculate(self, prev_state, next_state):
        """
        计算从 prev_state 转换到 next_state 所获得的奖励。
        :param prev_state: 动作执行前的游戏状态 (JSON/dict)
        :param next_state: 动作执行后的游戏状态 (JSON/dict)
        :return: 一个浮点数，代表总奖励值
        """
        total_reward = 0.0

        # 确保状态有效
        if prev_state is None or next_state is None:
            return 0.0

        # --- 1. 战斗内奖励 (只有在战斗中才计算) ---
        if prev_state.in_game and next_state.in_game and prev_state.in_combat:
            # 计算对敌人造成的总伤害
            prev_total_hp = sum(m.current_hp for m in prev_state.monsters)
            next_total_hp = sum(m.current_hp for m in next_state.monsters)
            damage_dealt = prev_total_hp - next_total_hp
            if damage_dealt > 0:
                total_reward += damage_dealt * self.DAMAGE_DEALT_MULTIPLIER

            # 计算自身受到的伤害
            if prev_state.player is not None and next_state.player is not None:
                damage_taken = prev_state.player.current_hp - next_state.player.current_hp
                if damage_taken > 0:
                    total_reward += damage_taken * self.DAMAGE_TAKEN_MULTIPLIER

            # 检查是否浪费能量 (仅在结束回合时触发)
            if prev_state.turn < next_state.turn and prev_state.player is not None: # 这标志着一个回合的结束
                wasted_energy = prev_state.player.energy
                if wasted_energy > 0:
                    total_reward += wasted_energy * self.WASTE_ENERGY_PENALTY

        # --- 2. 战斗结果奖励 ---
        # 赢得战斗: 从战斗状态进入非战斗状态
        if prev_state.in_combat and not next_state.in_combat:
            # 检查是否是最终BOSS战胜利
            if prev_state.floor == 55:
                total_reward += self.WIN_FINAL_BOSS_REWARD
            else:
                total_reward += self.WIN_BATTLE_REWARD
                # 检查是否是幕BOSS战胜利
                if prev_state.floor == 17:
                    total_reward += self.WIN_ACT1_BOSS_BONUS
                elif prev_state.floor == 34:
                    total_reward += self.WIN_ACT2_BOSS_BONUS
                elif prev_state.floor == 51:
                    total_reward += self.WIN_ACT3_BOSS_BONUS
        
        # 输掉战斗: 游戏结束
        if prev_state.in_game and not next_state.in_game:
            total_reward += self.LOSE_BATTLE_REWARD

        # --- 3. 游戏进程奖励 ---
        floor_change = next_state.floor - prev_state.floor
        if floor_change > 0:
            current_floor = prev_state.floor
            if 1 <= current_floor <= 17:
                total_reward += floor_change * self.FLOOR_INCREASE_ACT1
            elif 18 <= current_floor <= 34:
                total_reward += floor_change * self.FLOOR_INCREASE_ACT2
            elif 35 <= current_floor <= 51:
                total_reward += floor_change * self.FLOOR_INCREASE_ACT3
            else: # 52层及以上
                total_reward += floor_change * self.FLOOR_INCREASE_ACT4

        return total_reward