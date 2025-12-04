# 此Action非彼Action，和communication的Action区分开，此处定义的是AI agent的动作结构
from enum import Enum
from dataclasses import dataclass  # 简化Class定义，自动生成__init__等方法
from spirecomm.ai.constants import MAX_DECK_SIZE, MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_POTION_COUNT
from typing import Union


class DecomposedActionType(Enum):
    """分解式动作类型，作为模型的主输出头"""
    PLAY = 0
    CHOOSE = 1
    POTION_USE = 2
    POTION_DISCARD = 3
    END = 4
    PROCEED = 5
    CONFIRM = 6
    RETURN = 7
    SKIP = 8
    LEAVE = 9


    def to_action_type(self):
        """将分解式动作类型转换为基础动作类型"""
        mapping = {
            DecomposedActionType.END: ActionType.END, # type: ignore
            DecomposedActionType.PROCEED: ActionType.PROCEED, # type: ignore
            DecomposedActionType.RETURN: ActionType.RETURN, # type: ignore
            DecomposedActionType.SKIP: ActionType.SKIP, # type: ignore
            DecomposedActionType.CONFIRM: ActionType.CONFIRM, # type: ignore
            DecomposedActionType.LEAVE: ActionType.LEAVE, # type: ignore
            # 注意：PLAY, CHOOSE, POTION_USE, POTION_DISCARD 是有参数的，不应在这里转换
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
    decomposed_type: DecomposedActionType = DecomposedActionType.POTION_DISCARD
    def __post_init__(self):
        if not (0 <= self.potion_idx < MAX_POTION_COUNT): # type: ignore
            raise ValueError(f"Potion丢弃索引{self.potion_idx}超出范围（0~{MAX_POTION_COUNT-1}）")
    def to_string(self) -> str:
        # 格式："potion discard X"（X为药水索引，0-based）
        return f"potion discard {self.potion_idx}"

@dataclass(frozen=True)
class PotionUseAction(BaseAction):
    potion_idx: int  # 0-based
    target_idx: Union[int, None]  # None=无目标
    decomposed_type: DecomposedActionType = DecomposedActionType.POTION_USE
    def __post_init__(self):
        if not (0 <= self.potion_idx < MAX_POTION_COUNT): # type: ignore
            raise ValueError(f"Potion使用索引{self.potion_idx}超出范围（0~{MAX_POTION_COUNT-1}）")
        if self.target_idx is not None and not (0 <= self.target_idx < MAX_MONSTER_COUNT): # type: ignore
            raise ValueError(f"Potion目标索引{self.target_idx}超出范围（0~{MAX_MONSTER_COUNT-1}）")
    def to_string(self) -> str:
        # 格式1：无目标 → "potion use X"（X为药水索引）
        # 格式2：有目标 → "potion use X Y"（X=药水索引，Y=怪物索引）
        if self.target_idx is None:
            return f"potion use {self.potion_idx}"
        return f"potion use {self.potion_idx} {self.target_idx}"

@dataclass(frozen=True)
class SingleAction(BaseAction):
    # SingleAction 代表那些不带额外参数的动作（如 return/end/...）
    # 在构造时需要显式传入对应的 decomposed_type，例如:
    # SingleAction(type=ActionType.RETURN, decomposed_type=DecomposedActionType.RETURN)
    decomposed_type: DecomposedActionType
    def to_string(self) -> str:
        # 单一动作直接返回枚举值（如"return"、"end"，与你的原逻辑一致）
        return self.type.value