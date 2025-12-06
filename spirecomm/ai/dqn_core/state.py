from dataclasses import dataclass, field
from spirecomm.ai.constants import MAX_CHOOSE_COUNT, MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_POTION_COUNT
from spirecomm.spire.game import Game
from spirecomm.ai.dqn_core.action import  BaseAction, PlayAction, ChooseAction, PotionDiscardAction, PotionUseAction, SingleAction, ActionType, DecomposedActionType
from typing import List
import numpy as np
from spirecomm.ai.absolute_logger import AbsoluteLogger, LogType
import json

@dataclass
class GameStateProcessor:
    """
    游戏状态预处理器。
    它的主要职责是将 `Game` 对象转换为一个扁平化的 PyTorch 张量 (tensor)，
    以便作为神经网络的输入。
    这个实现将具体的向量化逻辑委托给 `Game` 对象自身的 `get_vector` 方法。
    """
    absolute_logger: AbsoluteLogger = field(default_factory=lambda: AbsoluteLogger(LogType.STATE))
    def __post_init__(self):
        self.absolute_logger.start_episode()
    def get_state_tensor(self, game: Game):
        """
        从 Game 对象获取完整的、扁平化的状态向量。
        :param game: spirecomm.spire.game.Game 的实例
        :return: 一个一维的 PyTorch 张量
        """
        return game.get_vector()

    def process(self, game: Game):
        """将原始 game_state 字典转换为 PyTorch 张量。"""
        # game_state 已经是 Game 对象了，直接使用
        vector_tensor = game.get_vector()
        # vector_tensor 已经是 Tensor，无需 from_numpy，只需确保类型和维度正确
        return vector_tensor.float().unsqueeze(0)

    def get_action_masks(self, game_state: Game):
        """
        根据当前游戏状态，生成所有分解动作的合法性掩码。
        """
        available_actions = self.get_available_actions(game_state)
        ava_commands = game_state.available_commands
        # 移除"key","click","wait","state"等无实际意义的命令
        ava_commands = [cmd for cmd in ava_commands if cmd not in ["key", "click", "wait", "state"]]
        self.absolute_logger.write(f"\n可用命令列表: {ava_commands}\n")
        self.absolute_logger.write(f"筛选之后结果: {[action.to_string() for action in available_actions]}\n")

        # 初始化所有掩码为 False
        action_type_mask = np.zeros(len(DecomposedActionType), dtype=bool)
        play_card_mask = np.zeros(MAX_HAND_SIZE, dtype=bool)
        target_monster_mask = np.zeros(MAX_MONSTER_COUNT, dtype=bool)
        choose_option_mask = np.zeros(MAX_CHOOSE_COUNT, dtype=bool)
        # 将药水掩码拆分为使用（use）和丢弃（discard）两路
        potion_use_mask = np.zeros(MAX_POTION_COUNT, dtype=bool)
        potion_discard_mask = np.zeros(MAX_POTION_COUNT, dtype=bool)

        for action in available_actions:
            # 有对应的 DecomposedActionType 才设置掩码
            if hasattr(action, 'decomposed_type'):
                decomposed_type:DecomposedActionType = action.decomposed_type
                type_val = int(decomposed_type.value)
                action_type_mask[type_val] = True

            if isinstance(action, PlayAction):
                play_card_mask[action.hand_idx] = True
                if action.target_idx is not None:
                    target_monster_mask[action.target_idx] = True
            elif isinstance(action, ChooseAction):
                choose_option_mask[action.choice_idx] = True
            elif isinstance(action, PotionUseAction):
                # 明确标记可 use 的药水位
                potion_use_mask[action.potion_idx] = True
                if action.target_idx is not None:
                    target_monster_mask[action.target_idx] = True
            elif isinstance(action, PotionDiscardAction):
                # 明确标记可 discard 的药水位
                potion_discard_mask[action.potion_idx] = True
                

        self.absolute_logger.write("动作掩码生成完毕。\n")
        self.absolute_logger.write({
            'action_mask':  str(action_type_mask.tolist()),
            'play_mask': str(play_card_mask.tolist()),
            'target_mask': str(target_monster_mask.tolist()),
            'choose_mask': str(choose_option_mask.tolist()),
            'potion_use_mask': str(potion_use_mask.tolist()),
            'potion_discard_mask': str(potion_discard_mask.tolist()),
            # 兼容：合并一个总的 potion_mask（use 或 discard 任一可行）
            'potion_mask': str((potion_use_mask | potion_discard_mask).tolist()),
            'hand': str([str(card.name) for card in game_state.hand]),
        })
        # 返回时提供独立的 potion_use / potion_discard，并保留向后兼容的 'potion'（合并）
        return {
            'action_type': action_type_mask,
            'play_card': play_card_mask,
            'target_monster': target_monster_mask,
            'choose_option': choose_option_mask,
            'potion_use': potion_use_mask,
            'potion_discard': potion_discard_mask,
            'potion': (potion_use_mask | potion_discard_mask)
        }

    def get_available_actions(self, game: Game) -> List[BaseAction]:
        """从 game_state 解析出所有合法的结构化动作对象列表"""
        actions = []
        
        # choose 动作
        if game.choice_available and "choose" in game.available_commands:
            for i, _ in enumerate(game.choice_list):
                actions.append(ChooseAction(type=ActionType.CHOOSE, choice_idx=i, decomposed_type=DecomposedActionType.CHOOSE))

        # 战斗中的动作
        if "play" in game.available_commands:
            # Play a card
            for hand_idx, card in enumerate(game.hand):
                if card.is_playable:
                    if card.has_target:
                        # 怪物列表直接在 game 对象下
                        for monster_idx, monster in enumerate(game.monsters):
                            if not monster.is_gone:
                                # 卡牌索引是它在手牌中的位置
                                actions.append(PlayAction(type=ActionType.PLAY, hand_idx=hand_idx, target_idx=monster_idx, decomposed_type=DecomposedActionType.PLAY))
                    else:
                        actions.append(PlayAction(type=ActionType.PLAY, hand_idx=hand_idx, target_idx=None, decomposed_type=DecomposedActionType.PLAY))
        
        # End turn
        if "end" in game.available_commands:
            actions.append(SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END))

        if "potion" in game.available_commands:
                for potion_idx, potion in enumerate(game.potions):
                    if potion.can_use:
                        if potion.requires_target:
                            for monster_idx, monster in enumerate(game.monsters):
                                if not monster.is_gone:
                                    # 药水索引是它在药水栏中的位置
                                    actions.append(PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=monster_idx, decomposed_type=DecomposedActionType.POTION_USE))
                        else:
                            actions.append(PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=None, decomposed_type=DecomposedActionType.POTION_USE))
                    if potion.can_discard:
                        actions.append(PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx, decomposed_type=DecomposedActionType.POTION_DISCARD))
        # 非战斗中的通用动作
        # 正确的判断方式是检查 available_commands
        if "confirm" in game.available_commands:
            actions.append(SingleAction(type=ActionType.CONFIRM, decomposed_type=DecomposedActionType.CONFIRM))
        # if "return" in game.available_commands:
        #     actions.append(SingleAction(type=ActionType.RETURN, decomposed_type=DecomposedActionType.RETURN))
        if "proceed" in game.available_commands:
            actions.append(SingleAction(type=ActionType.PROCEED, decomposed_type=DecomposedActionType.PROCEED))
        if "skip" in game.available_commands:
            actions.append(SingleAction(type=ActionType.SKIP, decomposed_type=DecomposedActionType.SKIP))
        if "leave" in game.available_commands:
            actions.append(SingleAction(type=ActionType.LEAVE, decomposed_type=DecomposedActionType.LEAVE))
        
        return actions