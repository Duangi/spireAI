from dataclasses import dataclass, field
import torch
import torch.nn as nn
from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized
@dataclass
class Relic:
    relic_id: str = field(default="")
    name: str = field(default="")
    counter: int = field(default=0)
    price: int = field(default=0)
    @classmethod
    def get_vec_length(self):
        return 3
    def get_vector(self):
        """将遗物属性转换为固定长度的向量表示"""
        vec = torch.tensor([
            minmax_normalize(self.counter, -1, 10),
        ],dtype=torch.float32)
        # 设计一个one-hot 如果counter是负数，表示没有计数功能
        has_counter_vec = torch.tensor([
            1.0 if self.counter >= 0 else 0.0
        ], dtype=torch.float32)
        # 添加名称的hash值作为特征
        hash = get_hash_val_normalized(self.relic_id + self.name)
        vec = torch.cat([vec, has_counter_vec, hash])
        return vec
    @classmethod
    def from_json(cls, json_object):
        return cls(json_object["id"], json_object["name"], json_object["counter"], json_object.get("price", 0))
    def __eq__(self, other):
        return other.relic_id == self.relic_id
if __name__ == "__main__":
    # 测试遗物向量表示
    relic = Relic(
        relic_id="relic_001",
        name="Ancient Coin",
        counter=5
    )
    vec = relic.get_vector()
    print(f"Relic Vector Shape: {vec.shape}")
    print(vec)