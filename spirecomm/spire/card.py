from enum import Enum
import torch
import torch.nn as nn
import json
import os

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


class Card:
    def __init__(self, card_id, name, card_type, rarity, upgrades=0, has_target=False, cost=0, uuid="", misc=0, price=0, is_playable=False, exhausts=False):
        self.card_id = card_id
        self.name = name
        self.type = card_type
        self.rarity = rarity
        self.upgrades = upgrades
        self.has_target = has_target
        self.cost = cost
        self.uuid = uuid
        self.misc = misc
        self.price = price
        self.is_playable = is_playable
        self.exhausts = exhausts

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
            price=json_object.get("price", 0),
            is_playable=json_object.get("is_playable", False),
            exhausts=json_object.get("exhausts", False)
        )

    def __eq__(self, other):
        return self.uuid == other.uuid
    
    

class CardManager:
    def __init__(self, filename="cards.json"):
        self.filename = os.path.join(os.path.dirname(__file__),filename)
        self.cards = {}
        
        # 从文件中读取max_index
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.cards = json.load(f)
        else:
            # 打开文件并往里面存储self.cards
            with open(self.filename, 'w') as f:
                json.dump(self.cards, f)
    def get_abspath(self):
        path = os.path.abspath(__file__)
        # 获取当前脚本所在的目录
        dir = os.path.dirname(path)
        return dir

    def get_card_index(self, card:Card):
        if card.card_id in self.cards:
            return self.cards[card.card_id]
        else:
            return self.add_card(card)

    # 做一个只能往里加卡牌的功能
    def add_card(self, card:Card):
        # 判断self.cards里面是否有和card.uuid相同的key
        if card.card_id in self.cards:
            return self.cards[card.card_id]
        else:
            # 如果没有，那么就把card.uuid加进去
            self.cards[card.card_id] = len(self.cards)
            # 再把self.cards写入文件
            with open( self.filename, 'w') as f:
                json.dump(self.cards, f)
            return self.cards[card.card_id]
        
    def get_card_embedding_vector(self, card:Card):
        # id是字符串，因此需要使用embedding层转化为向量
        # 游戏中大约有400张卡牌，因此embedding的维度可以设置为20
        embedding_layer = nn.Embedding(num_embeddings=500, embedding_dim=20)
        
        index = self.get_card_index(card)
        index_tensor = torch.tensor([int(index)], dtype=torch.long)
        index_vector = embedding_layer(index_tensor)
        # 将一些卡牌的属性转化为向量
        ntype = self.normalize(int(card.type.value), max=5)
        nrarity = self.normalize(int(card.rarity.value), max=6)
        nupgrades = self.normalize(int(card.upgrades), max=1)
        nhas_target = self.normalize(int(card.has_target), max=1)
        ncost = self.normalize(int(card.cost), max=3)
        # 将index_vector和fixed_vector合并
        fixed_vector = torch.tensor([ntype, nrarity, nupgrades, nhas_target, ncost], dtype=torch.float32)
        index_vector = index_vector.view(-1)
        # fixed_vector = tf.expand_dims(fixed_vector, axis=0)
        embedded_vector = torch.cat([index_vector, fixed_vector], dim=0)
        
        return embedded_vector
    
    # 将数据映射到0-1之间
    def normalize(self, x, max):
        return x / (max)