import torch.nn as nn
import torch
from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized
from dataclasses import dataclass, field
@dataclass
class Potion:
    potion_id: str = field(default="")
    name: str = field(default="")
    can_use: bool = field(default=False)
    can_discard: bool = field(default=False)
    requires_target: bool = field(default=False)

    @classmethod
    def get_vec_length(self):
        return 4
    def get_vector(self):
        """返回药水的向量表示"""
        bool_vec = torch.tensor([
            int(self.can_use),
            int(self.can_discard),
            int(self.requires_target)
        ], dtype=torch.float32)
        # 加入名称的hash值
        name_hash = get_hash_val_normalized(self.potion_id + self.name)
        vec = torch.cat([bool_vec, name_hash])
        return vec

    def __eq__(self, other):
        return other.potion_id == self.potion_id

    @classmethod
    def from_json(cls, json_object):
        return cls(
            potion_id=json_object.get("id"),
            name=json_object.get("name"),
            can_use=json_object.get("can_use", False),
            can_discard=json_object.get("can_discard", False),
            requires_target=json_object.get("requires_target", False)
        )
if __name__ == "__main__":
    # 测试药水向量表示
    potion = Potion(
        potion_id="potion_001",
        name="Healing Potion",
        can_use=True,
        can_discard=False,
        requires_target=False
    )
    vec = potion.get_vector()
    print(f"Potion Vector Shape: {vec.shape}")
    print(vec)