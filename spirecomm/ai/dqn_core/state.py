import torch
from dataclasses import dataclass, field
from spirecomm.spire.game import Game
from spirecomm.ai.dqn_core.action import ActionMapper, BaseAction, PlayAction, ChooseAction, PotionUseAction, SingleAction, ActionType, DecomposedActionType, MAX_MONSTER_COUNT
from typing import List
import numpy as np

@dataclass
class GameStateProcessor:
    """
    游戏状态预处理器。
    它的主要职责是将 `Game` 对象转换为一个扁平化的 PyTorch 张量 (tensor)，
    以便作为神经网络的输入。
    这个实现将具体的向量化逻辑委托给 `Game` 对象自身的 `get_vector` 方法。
    """

    def __post_init__(self):
        self.action_mapper = ActionMapper()

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
        
        # 初始化所有掩码为 False
        action_type_mask = np.zeros(len(DecomposedActionType), dtype=bool)
        play_card_mask = np.zeros(self.action_mapper.max_play_dim, dtype=bool)
        target_monster_mask = np.zeros(MAX_MONSTER_COUNT, dtype=bool)
        choose_option_mask = np.zeros(self.action_mapper.max_choose_dim, dtype=bool)
        potion_mask = np.zeros(self.action_mapper.max_potion_dim, dtype=bool)

        for action in available_actions:
            if hasattr(action, 'decomposed_type'):
                action_type_mask[action.decomposed_type.value] = True

            if isinstance(action, PlayAction):
                play_card_mask[action.hand_idx] = True
                if action.target_idx is not None:
                    target_monster_mask[action.target_idx] = True
            elif isinstance(action, ChooseAction):
                choose_option_mask[action.choice_idx] = True
            elif isinstance(action, PotionUseAction):
                potion_mask[action.potion_idx] = True
                if action.target_idx is not None:
                    target_monster_mask[action.target_idx] = True
        
        return {
            'action_type': action_type_mask,
            'play_card': play_card_mask,
            'target_monster': target_monster_mask,
            'choose_option': choose_option_mask,
            'potion': potion_mask
        }

    def get_available_actions(self, game: Game) -> List[BaseAction]:
        """从 game_state 解析出所有合法的结构化动作对象列表"""
        actions = []
        
        if game.choice_available:
            # 你是对的！选项列表是 game.choice_list
            for i in range(len(game.choice_list)):
                actions.append(ChooseAction(type=ActionType.CHOOSE, choice_idx=i, decomposed_type=DecomposedActionType.CHOOSE))

        # 战斗中的动作
        if game.in_combat:
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
            # Use a potion
            for potion_idx, potion in enumerate(game.potions):
                if potion.can_use:
                    if potion.requires_target:
                        for monster_idx, monster in enumerate(game.monsters):
                            if not monster.is_gone:
                                # 药水索引是它在药水栏中的位置
                                actions.append(PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=monster_idx, decomposed_type=DecomposedActionType.POTION))
                    else:
                        actions.append(PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=None, decomposed_type=DecomposedActionType.POTION))
            # End turn
            actions.append(SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END))

        # 非战斗中的通用动作
        # 正确的判断方式是检查 available_commands
        if "proceed" in game.available_commands:
            actions.append(SingleAction(type=ActionType.PROCEED, decomposed_type=DecomposedActionType.PROCEED))
        if "cancel" in game.available_commands:
            actions.append(SingleAction(type=ActionType.CANCEL, decomposed_type=DecomposedActionType.CANCEL))
        
        return actions

    def get_start_game_action(self, game: Game) -> str:
        """获取开始游戏的动作"""
        if "start" in game.available_commands:
            # 角色信息在 game.character 中
            return "start " + game.character.name.lower()
        else:
            return "proceed"
