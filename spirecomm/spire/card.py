from enum import Enum
import torch
import torch.nn as nn

import xxhash
import numpy as np
from dataclasses import dataclass, field
from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized

MAX_UINT64 = 2**64 - 1

class CardType(Enum):
    ATTACK = 1
    SKILL = 2
    POWER = 3
    STATUS = 4
    CURSE = 5

class CardRarity(Enum):
    BASIC = 1
    COMMON = 2
    UNCOMMON = 3
    RARE = 4
    SPECIAL = 5
    CURSE = 6

@dataclass
class Card:
    card_id: str = field(default="")
    name: str = field(default="")
    card_type: CardType = field(default=CardType.ATTACK)
    rarity: CardRarity = field(default=CardRarity.COMMON)
    upgrades: int = field(default=0)
    has_target: bool = field(default=False)
    cost: int = field(default=0)
    uuid: str = field(default="")
    misc: int = field(default=0)
    is_playable: bool = field(default=False)
    exhausts: bool = field(default=False)

    

    def __eq__(self, other):
        return self.uuid == other.uuid
    
    # 将数据映射到0-1之间
    def normalize(self, x, max):
        return x / (max)
    
    def get_vector(self):
        """将卡牌属性转换为固定长度的向量表示"""
        # 1. 连续特征（已归一化）
        continuous_vec = torch.tensor([
            self.card_level_normalize_piecewise(self.upgrades),
            minmax_normalize(self.cost, 0, 10),
        ], dtype=torch.float32)

        # 2. 多类别离散特征（需one-hot编码，类别数≥3）
        one_hot_card_type = nn.functional.one_hot(
            torch.tensor(self.card_type.value - 1, dtype=torch.long),  # 确保long类型
            num_classes=5
        ).float()  # 转float与其他向量一致
        one_hot_rarity = nn.functional.one_hot(
            torch.tensor(self.rarity.value - 1, dtype=torch.long),
            num_classes=6
        ).float()

        # 3. bool类型特征（1维0/1编码，替代2维one-hot）
        bool_vec = torch.tensor([
            int(self.has_target),
            int(self.is_playable),
            int(self.exhausts)
        ], dtype=torch.float32)

        # 4. uuid hash特征
        hash_vec = get_hash_val_normalized(self.uuid)

        # 一次性拼接所有向量（减少重复cat操作，更高效）
        final_vec = torch.cat([
            continuous_vec,
            one_hot_card_type,
            one_hot_rarity,
            bool_vec,
            hash_vec
        ])

        return final_vec
    @classmethod
    def get_vec_length(self):
        return 17
    @classmethod
    def from_json(cls, json_object):
        return cls(
            card_id=json_object["id"],
            name=json_object["name"],
            card_type=CardType[json_object["type"]],
            rarity=CardRarity[json_object["rarity"]],
            upgrades=json_object["upgrades"],
            has_target=json_object["has_target"],
            cost=json_object["cost"],
            uuid=json_object["uuid"],
            misc=json_object.get("misc", 0),
            is_playable=json_object.get("is_playable", False),
            exhausts=json_object.get("exhausts", False)
        )
    
    def card_level_normalize_piecewise(self, x) -> torch.Tensor:
        """
        分段函数:x=1→y=0.5,前期快涨，后期边际递减
        段1(0≤x≤1):线性快涨(0→0.5)
        段2(1<x≤5):指数快涨(0.5→0.85)
        段3(x>5):对数缓涨(0.85→1.0)
        """
        # 确保输入转为PyTorch标量张量（支持梯度）
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        assert x.ndim == 0, "输入必须是标量（单个数值），不能是批量张量"
        
        # 分段逻辑（标量专属，简洁高效，可导）
        if x <= 1:
            # 段1（0≤x≤1）：线性快涨，x=1→y=0.5
            y = 0.5 * x
        elif 1 < x <= 5:
            # 段2（1<x≤5）：指数快涨，x=5→y=0.85
            y = 0.5 + 0.35 * (1 - torch.exp(-0.8 * (x - 1)))
        else:
            # 段3（x>5）：对数缓涨，趋近于1，边际递减
            y = 0.85 + 0.15 * (torch.log1p(x - 5) / torch.log1p(torch.tensor(25.0)))
        
        # 限制值域在[0,1]，避免极端值
        return torch.clamp(y, 0.0, 1.0)
if __name__ == "__main__":
    # 测试vector输出
    card = Card(
        card_id="test_card",
        name="Test Card",
        card_type=CardType.ATTACK,
        rarity=CardRarity.COMMON,
        upgrades=3,
        has_target=False,
        cost=2,
        uuid="test_uuid_12345",
        misc=0,
        is_playable=False,
        exhausts=False
    )
    vec = card.get_vector()
    print(f"Card Vector Shape: {vec.shape}")