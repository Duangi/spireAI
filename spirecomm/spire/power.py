import spirecomm.spire.card
import torch
from spirecomm.spire.card import Card 
from spirecomm.utils.data_processing import minmax_normalize, normal_normalize, get_hash_val_normalized
class Power:

    def __init__(self, power_id, name, amount, damage=0, misc=0, just_applied=False, card=None):
        self.power_id:str = power_id
        self.power_name:str = name
        self.amount:int = amount # 层数
        self.damage:int = damage # 伤害，大部分buff应该没有伤害的
        self.misc:int = misc # ？？
        self.just_applied:bool = just_applied # 这又是什么
        self.card:Card = card # 怎么还有张card
    @classmethod
    def get_vec_length(self):
        return 4

    def get_vector(self):
        amount_tensor = torch.tensor([normal_normalize(self.amount,0,50)], dtype=torch.float32)
        damage_tensor = torch.tensor([normal_normalize(self.damage,0,20)], dtype=torch.float32)
        bool_tensor = torch.tensor([int(self.just_applied)], dtype=torch.float32)
        id_tensor = torch.tensor([get_hash_val_normalized(self.power_id + self.power_name)], dtype=torch.float32)

        final_vec = torch.cat([amount_tensor,damage_tensor,bool_tensor,id_tensor])
        return final_vec

    @classmethod
    def from_json(cls, json_object):
        power_id = json_object["id"]
        name = json_object["name"]
        amount = json_object["amount"]
        damage = json_object.get("damage", 0)
        misc = json_object.get("misc", 0)
        just_applied = json_object.get("just_applied", False)
        card = json_object.get("card", None)
        if card is not None:
            card = spirecomm.spire.card.Card.from_json(card)
        return cls(power_id, name, amount, damage, misc, just_applied, card)

    def __eq__(self, other):
        return self.power_id == other.power_id and self.amount == other.amount
