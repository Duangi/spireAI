from typing import Tuple
import torch.nn as nn
import torch
from spirecomm.utils.data_processing import get_hash_id, minmax_normalize, get_hash_val_normalized
from dataclasses import dataclass, field
@dataclass
class Potion:
    potion_id: str = field(default="")
    name: str = field(default="")
    can_use: bool = field(default=False)
    can_discard: bool = field(default=False)
    requires_target: bool = field(default=False)
    price: int = field(default=0)

    def get_tensor_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回药水的结构化数据。
        
        Returns:
            emb_id: LongTensor (用于查万能字典)
            feats:  FloatTensor (数值特征: CanUse, CanDiscard, RequiresTarget, Price)
        """
        # 1. 身份 ID (用于 Embedding)
        # 使用 name 作为唯一标识 , 后续ChoiceList有用
        emb_id = torch.tensor(get_hash_id(self.name), dtype=torch.long)

        # 2. 数值特征 (用于拼接)
        can_use_val = 1.0 if self.can_use else 0.0
        can_discard_val = 1.0 if self.can_discard else 0.0
        requires_target_val = 1.0 if self.requires_target else 0.0
        price_val = minmax_normalize(self.price, 0.0, 200.0)  # 假设最大价格200

        # 拼接数值特征 (4维)
        feats = torch.tensor([can_use_val, can_discard_val, requires_target_val, price_val], dtype=torch.float32)

        return emb_id, feats
    def __eq__(self, other):
        return other.potion_id == self.potion_id

    @classmethod
    def from_json(cls, json_object):
        return cls(
            potion_id=json_object.get("id"),
            name=json_object.get("name"),
            can_use=json_object.get("can_use", False),
            can_discard=json_object.get("can_discard", False),
            requires_target=json_object.get("requires_target", False),
            price=json_object.get("price", 0)
        )