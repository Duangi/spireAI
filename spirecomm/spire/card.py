from enum import Enum
from typing import Tuple
import torch
import torch.nn as nn

import xxhash
import numpy as np
from dataclasses import dataclass, field
from spirecomm.utils.data_processing import get_hash_id, minmax_normalize, get_hash_val_normalized

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
    
    def get_tensor_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回适配新架构的 (Embedding_ID, Numeric_Features)
        """
        
        # ==========================================
        # 1. 生成 ID (用于查万能字典)
        # ==========================================
        # 注意：这里 Hash 的是 name,因为choice_list里出现卡牌时，是以name为准的。
        emb_id = torch.tensor(get_hash_id(self.name), dtype=torch.long)

        # ==========================================
        # 2. 生成手工数值特征 (用于拼接)
        # ==========================================
        
        # A. 连续特征
        # 升级次数
        upgrade_feat = self.card_level_normalize_piecewise(self.upgrades)
        # 费用 (归一化)
        cost_feat = minmax_normalize(self.cost, 0, 5) # 费用很少超过5

        continuous_vec = torch.tensor([upgrade_feat, cost_feat], dtype=torch.float32)

        # B. 离散特征 (One-hot)
        # 这种显式的类型信息对模型冷启动非常有帮助，补充 Embedding 还没学好的时候
        one_hot_card_type = nn.functional.one_hot(
            torch.tensor(self.card_type.value - 1, dtype=torch.long),
            num_classes=len(CardType)
        ).float()
        
        one_hot_rarity = nn.functional.one_hot(
            torch.tensor(self.rarity.value - 1, dtype=torch.long),
            num_classes=len(CardRarity)
        ).float()

        # C. 布尔特征
        bool_vec = torch.tensor([
            1.0 if self.has_target else 0.0,
            1.0 if self.is_playable else 0.0,
            1.0 if self.exhausts else 0.0
        ], dtype=torch.float32)

        # D. 拼接所有数值特征 (移除了 UUID Hash)
        # 维度计算: 2 (连续) + 5 (Type) + 6 (Rarity) + 3 (Bool) = 16 维
        features_vec = torch.cat([
            continuous_vec,
            one_hot_card_type,
            one_hot_rarity,
            bool_vec
        ])

        return emb_id, features_vec
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