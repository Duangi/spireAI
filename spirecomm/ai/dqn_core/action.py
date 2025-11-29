# 此Action非彼Action，和communication的Action区分开，此处定义的是AI agent的动作结构
from enum import Enum
from dataclasses import dataclass  # 简化Class定义，自动生成__init__等方法
from spirecomm.ai.constants import MAX_DECK_SIZE, MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_POTION_COUNT
from typing import Union


class DecomposedActionType(Enum):
    """分解式动作类型，作为模型的主输出头"""
    PLAY = 0
    CHOOSE = 1
    END = 2
    POTION = 3
    PROCEED = 4
    CANCEL = 5

    def to_action_type(self):
        """将分解式动作类型转换为基础动作类型"""
        mapping = {
            DecomposedActionType.END: ActionType.END,
            DecomposedActionType.PROCEED: ActionType.PROCEED,
            DecomposedActionType.CANCEL: ActionType.CANCEL,
            # 注意：PLAY, CHOOSE, POTION 是有参数的，不应在这里转换
        }
        return mapping.get(self)

NUM_ACTION_TYPES = len(DecomposedActionType)

# 动作类型枚举（合并药水使用动作）
class ActionType(Enum):
    CHOOSE = "choose"
    PLAY = "play"
    POTION_DISCARD = "potion_discard"
    POTION_USE = "potion_use"
    RETURN = "return"
    END = "end"
    PROCEED = "proceed"
    SKIP = "skip"
    LEAVE = "leave"
    CONFIRM = "confirm"
    CANCEL = "cancel"

# 结构化动作Class（均添加 to_string() 方法）
@dataclass(frozen=True)
class BaseAction:
    type: ActionType
    def to_string(self) -> str:
        """转为游戏环境可执行的字符串动作（子类必须实现）"""
        raise NotImplementedError("子类必须实现 to_string 方法")

@dataclass(frozen=True)
class ChooseAction(BaseAction):
    choice_idx: int  # 0-based
    decomposed_type: DecomposedActionType = DecomposedActionType.CHOOSE
    def __post_init__(self):
        if not (0 <= self.choice_idx < MAX_DECK_SIZE):
            raise ValueError(f"Choose索引{self.choice_idx}超出范围（0~{MAX_DECK_SIZE-1}）")
    def to_string(self) -> str:
        # 格式："choose X"（X为选择项索引，0-based，与你的原逻辑一致）
        return f"choose {self.choice_idx}"

@dataclass(frozen=True)
class PlayAction(BaseAction):
    hand_idx: int  # 0-based（游戏状态中手牌索引）
    target_idx: Union[int, None]  # None=无目标
    decomposed_type: DecomposedActionType = DecomposedActionType.PLAY
    def __post_init__(self):
        if not (0 <= self.hand_idx < MAX_HAND_SIZE):
            raise ValueError(f"Play手牌索引{self.hand_idx}超出范围（0~{MAX_HAND_SIZE-1}）")
        if self.target_idx is not None and not (0 <= self.target_idx < MAX_MONSTER_COUNT):
            raise ValueError(f"Play目标索引{self.target_idx}超出范围（0~{MAX_MONSTER_COUNT-1}）")
    def to_string(self) -> str:
        # 格式1：无目标 → "play X"（X为1-based手牌索引，兼容你的原逻辑）
        # 格式2：有目标 → "play X Y"（X=1-based手牌索引，Y=怪物索引）
        hand_idx_1based = self.hand_idx + 1  # 转回1-based（和你原代码一致）
        if self.target_idx is None:
            return f"play {hand_idx_1based}"
        return f"play {hand_idx_1based} {self.target_idx}"

@dataclass(frozen=True)
class PotionDiscardAction(BaseAction):
    potion_idx: int  # 0-based
    def __post_init__(self):
        if not (0 <= self.potion_idx < MAX_POTION_COUNT):
            raise ValueError(f"Potion丢弃索引{self.potion_idx}超出范围（0~{MAX_POTION_COUNT-1}）")
    def to_string(self) -> str:
        # 格式："potion discard X"（X为药水索引，0-based）
        return f"potion discard {self.potion_idx}"

@dataclass(frozen=True)
class PotionUseAction(BaseAction):
    potion_idx: int  # 0-based
    target_idx: Union[int, None]  # None=无目标
    decomposed_type: DecomposedActionType = DecomposedActionType.POTION
    def __post_init__(self):
        if not (0 <= self.potion_idx < MAX_POTION_COUNT):
            raise ValueError(f"Potion使用索引{self.potion_idx}超出范围（0~{MAX_POTION_COUNT-1}）")
        if self.target_idx is not None and not (0 <= self.target_idx < MAX_MONSTER_COUNT):
            raise ValueError(f"Potion目标索引{self.target_idx}超出范围（0~{MAX_MONSTER_COUNT-1}）")
    def to_string(self) -> str:
        # 格式1：无目标 → "potion use X"（X为药水索引）
        # 格式2：有目标 → "potion use X Y"（X=药水索引，Y=怪物索引）
        if self.target_idx is None:
            return f"potion use {self.potion_idx}"
        return f"potion use {self.potion_idx} {self.target_idx}"

@dataclass(frozen=True)
class SingleAction(BaseAction):
    decomposed_type: DecomposedActionType
    def to_string(self) -> str:
        # 单一动作直接返回枚举值（如"return"、"end"，与你的原逻辑一致）
        return self.type.value
    
class ActionMapper:
    """动作映射器，将动作映射到索引范围
    根据当前的MAX_HAND_SIZE = 10, MAX_MONSTER_COUNT = 5, MAX_DECK_SIZE = 100, MAX_POTION_COUNT = 5
    计算出各类动作的索引范围，方便DQN输出动作索引的映射,总动作维度为 100 + 10 + 10*5  + 5 + 5*5 + 7 = 202
    """
    def __init__(self):
        """初始化动作映射器（完全保留你的逻辑）"""
        self.max_choose_dim = MAX_DECK_SIZE
        self.max_play_dim = MAX_HAND_SIZE + MAX_HAND_SIZE * MAX_MONSTER_COUNT
        self.max_potion_dim = MAX_POTION_COUNT + MAX_POTION_COUNT + MAX_POTION_COUNT * MAX_MONSTER_COUNT 
        self.max_single_dim = 7

        self.max_action_dim = self.max_choose_dim + self.max_play_dim + self.max_potion_dim + self.max_single_dim

        # 0-99: choose动作
        self.choose_start = 0
        self.choose_end = self.choose_start + self.max_choose_dim - 1

        # 100 - 109: play无目标动作
        # 110 - 159: play有目标动作
        self.play_start = self.choose_end + 1
        self.play_without_target_start = self.choose_end + 1
        self.play_without_target_end = self.play_without_target_start + MAX_HAND_SIZE - 1
        self.play_with_target_start = self.play_without_target_end + 1
        self.play_with_target_end = self.play_with_target_start + (MAX_HAND_SIZE * MAX_MONSTER_COUNT) - 1
        self.play_end = self.play_start + self.max_play_dim - 1

        # 160 - 164: potion丢弃动作
        # 165 - 169: potion使用无目标动作
        # 170 - 194: potion使用有目标动作
        self.potion_start = self.play_end + 1
        self.potion_discard_start = self.play_end + 1
        self.potion_discard_end = self.potion_discard_start + MAX_POTION_COUNT - 1
        self.potion_use_without_target_start = self.potion_discard_end + 1
        self.potion_use_without_target_end = self.potion_use_without_target_start + MAX_POTION_COUNT - 1
        self.potion_use_with_target_start = self.potion_use_without_target_end + 1
        self.potion_use_with_target_end = self.potion_use_with_target_start + (MAX_POTION_COUNT * MAX_MONSTER_COUNT) - 1
        self.potion_end = self.potion_start + self.max_potion_dim - 1

        # 195 - 201: 单一动作
        self.single_start = self.potion_end + 1
        self.single_actions = {
            ActionType.RETURN: self.single_start + 0, 
            ActionType.END: self.single_start + 1,    
            ActionType.PROCEED: self.single_start + 2,
            ActionType.SKIP: self.single_start + 3,
            ActionType.LEAVE: self.single_start + 4,
            ActionType.CONFIRM: self.single_start + 5,
            ActionType.CANCEL: self.single_start + 6,
        }
        self.single_end = self.single_start + self.max_single_dim - 1

    def action_to_index(self, action):
        """结构化动作 → 全局索引"""
        if isinstance(action, ChooseAction):
            return self.choose_start + action.choice_idx
        elif isinstance(action, PlayAction):
            if action.target_idx is None:
                return self.play_without_target_start + action.hand_idx
            else:
                offset = action.hand_idx * MAX_MONSTER_COUNT + action.target_idx
                return self.play_with_target_start + offset
        elif isinstance(action, PotionDiscardAction):
            return self.potion_discard_start + action.potion_idx
        elif isinstance(action, PotionUseAction):
            if action.target_idx is None:
                return self.potion_use_without_target_start + action.potion_idx
            else:
                offset = action.potion_idx * MAX_MONSTER_COUNT + action.target_idx
                return self.potion_use_with_target_start + offset
        elif isinstance(action, SingleAction):
            if action.type not in self.single_actions:
                raise ValueError(f"未知单一动作类型：{action.type}")
            return self.single_actions[action.type]
        else:
            raise TypeError(f"不支持的动作类型：{type(action)}")

    def index_to_action(self, idx):
        """全局索引 → 结构化动作"""
        if self.choose_start <= idx <= self.choose_end:
            return ChooseAction(type=ActionType.CHOOSE, choice_idx=idx - self.choose_start)
        elif self.play_without_target_start <= idx <= self.play_without_target_end:
            hand_idx = idx - self.play_without_target_start
            return PlayAction(type=ActionType.PLAY, hand_idx=hand_idx, target_idx=None)
        elif self.play_with_target_start <= idx <= self.play_with_target_end:
            offset = idx - self.play_with_target_start
            hand_idx = offset // MAX_MONSTER_COUNT
            target_idx = offset % MAX_MONSTER_COUNT
            return PlayAction(type=ActionType.PLAY, hand_idx=hand_idx, target_idx=target_idx)
        elif self.potion_discard_start <= idx <= self.potion_discard_end:
            potion_idx = idx - self.potion_discard_start
            return PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx)
        elif self.potion_use_without_target_start <= idx <= self.potion_use_without_target_end:
            potion_idx = idx - self.potion_use_without_target_start
            return PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=None)
        elif self.potion_use_with_target_start <= idx <= self.potion_use_with_target_end:
            offset = idx - self.potion_use_with_target_start
            potion_idx = offset // MAX_MONSTER_COUNT
            target_idx = offset % MAX_MONSTER_COUNT
            return PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=target_idx)
        elif self.single_start <= idx <= self.single_end:
            for action_type, action_idx in self.single_actions.items():
                if action_idx == idx:
                    return SingleAction(type=action_type)
        else:
            raise ValueError(f"无效动作索引：{idx}（总维度{self.max_action_dim}，有效范围0~{self.max_action_dim - 1}）")
        