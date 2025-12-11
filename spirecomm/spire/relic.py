from dataclasses import dataclass, field
from typing import Tuple
import torch
import torch.nn as nn
from spirecomm.utils.data_processing import get_hash_id, minmax_normalize, get_hash_val_normalized, norm_linear_clip
@dataclass
class Relic:
    relic_id: str = field(default="")
    name: str = field(default="")
    counter: int = field(default=0)
    price: int = field(default=0)
    
    def get_tensor_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回遗物的结构化数据。
        
        Returns:
            emb_id: LongTensor (用于查万能字典)
            feats:  FloatTensor (数值特征: Counter, HasCounter, Price)
        """
        # 1. 身份 ID (用于 Embedding)
        # 使用 name 作为唯一标识 , 后续ChoiceList有用
        emb_id = torch.tensor(get_hash_id(self.name), dtype=torch.long)

        # 2. 数值特征 (用于拼接)
        # counter: 很多遗物有计数 (如 笔尖=10, 香炉=6). 归一化到 0-1.
        # -1 通常表示无计数，这里我们用 max(0) 处理
        counter_val = norm_linear_clip(self.counter, 10.0) 
        
        # has_counter: 明确告诉模型这个遗物是否有计数功能
        has_counter = 1.0 if self.counter >= 0 else 0.0
        
        # price: 主要在商店有用，平时为0也无所谓
        price_val = norm_linear_clip(self.price, 400.0)

        # 拼接数值特征 (3维)
        feats = torch.tensor([counter_val, has_counter, price_val], dtype=torch.float32)

        return emb_id, feats

    @classmethod
    def get_feature_dim(cls):
        """返回数值特征向量的长度 (3)"""
        return 3
    
    @classmethod
    def from_json(cls, json_object):
        return cls(json_object["id"], json_object["name"], json_object["counter"], json_object.get("price", 0))
    def __eq__(self, other):
        return other.relic_id == self.relic_id