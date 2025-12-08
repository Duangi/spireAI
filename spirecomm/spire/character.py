from dataclasses import dataclass
from enum import Enum
from typing import List
import logging

import torch
import torch.nn as nn
from spirecomm.spire.power import Power
from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized, normal_normalize
from spirecomm.ai.constants import MAX_ORB_COUNT, MAX_POWER_COUNT

class Intent(Enum):
    ATTACK = 1
    ATTACK_BUFF = 2
    ATTACK_DEBUFF = 3
    ATTACK_DEFEND = 4
    BUFF = 5
    DEBUFF = 6
    STRONG_DEBUFF = 7
    DEBUG = 8
    DEFEND = 9
    DEFEND_DEBUFF = 10
    DEFEND_BUFF = 11
    ESCAPE = 12
    MAGIC = 13
    NONE = 14
    SLEEP = 15
    STUN = 16
    UNKNOWN = 17

    def is_attack(self):
        return self in [Intent.ATTACK, Intent.ATTACK_BUFF, Intent.ATTACK_DEBUFF, Intent.ATTACK_DEFEND]


class PlayerClass(Enum):
    IRONCLAD = 1
    THE_SILENT = 2
    DEFECT = 3
    WATCHER = 4

    def get_chinese_name(self):
        if self == PlayerClass.IRONCLAD:
            return "铁血战士"
        elif self == PlayerClass.THE_SILENT:
            return "静默猎手"
        elif self == PlayerClass.DEFECT:
            return "故障机器人"
        elif self == PlayerClass.WATCHER:
            return "观者"
        else:
            return "未知职业"


class OrbType(Enum):
    LIGHTNING = 1
    FROST = 2
    DARK = 3
    PLASMA = 4

class Orb:

    def __init__(self, name, orb_id, evoke_amount, passive_amount):
        self.name = name
        self.orb_id = orb_id
        self.evoke_amount = evoke_amount # 激发
        self.passive_amount = passive_amount # 被动
    @classmethod
    def get_vec_length(self):
        return 6
    def get_vector(self):
        """将球的属性转换为固定长度的向量表示"""
        # 1. 对 orb_id 进行 one-hot 编码
        one_hot_id = nn.functional.one_hot(
            torch.tensor(self.orb_id.value - 1, dtype=torch.long),
            num_classes=len(OrbType)
        ).float()
        # 2. 对 evoke_amount 和 passive_amount 进行归一化
        normalized_amounts = torch.tensor([
            normal_normalize(self.evoke_amount, 0, 50),    # 假设最大值为20
            normal_normalize(self.passive_amount, 0, 20) # 假设最大值为20
        ], dtype=torch.float32)
        return torch.cat([one_hot_id, normalized_amounts])
    @classmethod
    def from_str_to_type(cls, id_str: str):
        """
        将字符串 id 转成 OrbType；遇到 None / 空 / 占位字符串（如 "empty"）或未知 id 时
        返回 None（上层会忽略该 orb），并写警告日志。
        """
        if id_str is None:
            return None
        s = str(id_str).strip()
        if s == "" or s.lower() == "empty":
            logging.getLogger(__name__).warning("orb id is empty/placeholder: %r", id_str)
            return None
        # 允许常见大小写形式直接映射
        key = s.lower()
        mapping = {
            "lightning": OrbType.LIGHTNING,
            "frost": OrbType.FROST,
            "dark": OrbType.DARK,
            "plasma": OrbType.PLASMA
        }
        if key in mapping:
            return mapping[key]
        # 兼容直接使用枚举名（不区分大小写）
        try:
            return OrbType[s.upper()]
        except Exception:
            logging.getLogger(__name__).warning("unknown orb id: %r, skipping orb", id_str)
            return None

    @classmethod
    def from_json(cls, json_object):
        """从 JSON 构造 Orb；若无法解析 orb_id 返回 None（上层会跳过）。"""
        name = json_object.get("name")
        orb_id = cls.from_str_to_type(json_object.get("id"))
        if orb_id is None:
            # 无效/占位 id，返回 None 以便调用方过滤
            return None
        evoke_amount = json_object.get("evoke_amount", 0)
        passive_amount = json_object.get("passive_amount", 0)
        orb = Orb(name, orb_id, evoke_amount, passive_amount)
        return orb


class Character:

    def __init__(self, max_hp, current_hp=None, block=0):
        self.max_hp = max_hp
        self.current_hp = current_hp
        if self.current_hp is None:
            self.current_hp = self.max_hp
        self.block = block # 护盾
        self.powers:List[Power] = [] # 能力


class Player(Character):
    def __init__(self, max_hp, current_hp=None, block=0, energy=0):
        super().__init__(max_hp, current_hp, block)
        self.energy:int = energy
        self.orbs = []
    @classmethod
    def get_vec_length(cls):
        # block, energy, orbs, powers
        return 1 + 1 + MAX_ORB_COUNT * Orb.get_vec_length() + MAX_POWER_COUNT * Power.get_vec_length()
    def get_vector(self):
        """将玩家的属性转换为固定长度的向量表示"""
        # 1. 归一化 block 和 energy
        block_tensor = torch.tensor([normal_normalize(self.block, 0, 50)], dtype=torch.float32)
        energy_tensor = torch.tensor([normal_normalize(self.energy, 0, 5)], dtype=torch.float32)

        # 2. 处理充能球(orbs)，并填充到 MAX_ORB_COUNT
        orb_vectors = [orb.get_vector() for orb in self.orbs]
        # 如果有充能球，将它们堆叠成一个2D张量
        if orb_vectors:
            orbs_tensor = torch.stack(orb_vectors)
        else:
            # 如果没有，创建一个空的2D张量以便后续填充
            orbs_tensor = torch.empty(0, Orb.get_vec_length())
        # 计算需要填充的零向量数量，并创建填充张量
        orb_padding_count = MAX_ORB_COUNT - len(orb_vectors)
        orb_padding = torch.zeros(orb_padding_count, Orb.get_vec_length())
        # 拼接并展平为1D向量
        orbs_tensor = torch.cat([orbs_tensor, orb_padding], dim=0).flatten()

        # 3. 处理能力(powers)，并填充到 MAX_POWER_COUNT (逻辑与orbs类似)
        power_vectors = [power.get_vector() for power in self.powers]
        if power_vectors:
            powers_tensor = torch.stack(power_vectors)
        else:
            powers_tensor = torch.empty(0, Power.get_vec_length())
        power_padding_count = MAX_POWER_COUNT - len(power_vectors)
        power_padding = torch.zeros(power_padding_count, Power.get_vec_length())
        powers_tensor = torch.cat([powers_tensor, power_padding], dim=0).flatten()

        # 4. 拼接所有特征向量
        return torch.cat([block_tensor, energy_tensor, orbs_tensor, powers_tensor])

    @classmethod
    def from_json(cls, json_object):
        player = cls(json_object["max_hp"], json_object["current_hp"], json_object["block"], json_object["energy"])
        player.powers = [Power.from_json(json_power) for json_power in json_object["powers"]]
        # 安全解析 orbs：跳过 id 为 None 的项；Orb.from_json 返回 None 时也过滤掉
        player.orbs = []
        for orb_json in json_object.get("orbs", []):
            # 跳过根本没有 id 字段的条目
            if orb_json is None:
                continue
            orb_obj = Orb.from_json(orb_json)
            if orb_obj is not None:
                player.orbs.append(orb_obj)
        return player
    
    def to_json(self):
        return {
            "max_hp": self.max_hp,
            "current_hp": self.current_hp,
            "block": self.block,
            "energy": self.energy
        }

class Monster(Character):

    def __init__(self, name, monster_id, max_hp, current_hp, block, intent, half_dead, is_gone, move_id=-1, last_move_id=None, second_last_move_id=None, move_base_damage=0, move_adjusted_damage=0, move_hits=0):
        super().__init__(max_hp, current_hp, block)
        self.name = name
        self.monster_id = monster_id 
        self.intent = intent # one-hot量化
        self.half_dead = half_dead # one-hot量化
        self.is_gone = is_gone # one-hot量化 
        self.move_id = move_id # 不需要量化。并不是很懂是啥意思，但是我认为只要有intent和move_adjusted_damage就够了
        self.last_move_id = last_move_id 
        self.second_last_move_id = second_last_move_id
        self.move_base_damage = move_base_damage # 正态分布归一化，最多应该就50吧
        self.move_adjusted_damage = move_adjusted_damage # 正态分布归一化，强化后也50好了
        self.move_hits = move_hits # 正态分布归一化，一般来说都是3以内
        self.monster_index = 0
    @classmethod
    def get_vec_length(self):
        return 23
    def get_vector(self):
        """将怪物属性转换为固定长度的向量表示（优化后）"""
        # 1. 多类别离散特征：Intent（需one-hot，类别数=len(Intent)）
        one_hot_intent = nn.functional.one_hot(
            torch.tensor(int(self.intent.value) - 1, dtype=torch.long),  # 输入必须是long类型
            num_classes=len(Intent)
        ).float()  # 转float，与其他特征类型一致

        # 2. bool类型特征：用1维0/1编码（替代2维one-hot）
        bool_vec = torch.tensor([
            int(self.half_dead),
            int(self.is_gone)
        ], dtype=torch.float32)

        # 3. 连续特征：正态分布归一化（统一打包为tensor）
        continuous_vec = torch.tensor([
            normal_normalize(self.move_base_damage, 0, 50),
            normal_normalize(self.move_adjusted_damage, 0, 50),
            normal_normalize(self.move_hits, 0, 3)
        ], dtype=torch.float32)

        # 4. hash特征：怪物ID+名称的归一化hash（直接转1维tensor）
        hash_vec = torch.tensor([get_hash_val_normalized(f"{self.monster_id}{self.name}")], dtype=torch.float32)

        # 一次性拼接所有特征（高效且清晰）
        final_vec = torch.cat([
            one_hot_intent,
            bool_vec,
            continuous_vec,
            hash_vec
        ])

        return final_vec
    @classmethod
    def from_json(cls, json_object):
        name = json_object["name"]
        monster_id = json_object["id"]
        max_hp = json_object["max_hp"]
        current_hp = json_object["current_hp"]
        block = json_object["block"]
        intent = Intent[json_object["intent"]]
        half_dead = json_object["half_dead"]
        is_gone = json_object["is_gone"]
        move_id = json_object.get("move_id", -1)
        last_move_id = json_object.get("last_move_id", None)
        second_last_move_id = json_object.get("second_last_move_id", None)
        move_base_damage = json_object.get("move_base_damage", 0)
        move_adjusted_damage = json_object.get("move_adjusted_damage", 0)
        move_hits = json_object.get("move_hits", 0)
        monster = cls(name, monster_id, max_hp, current_hp, block, intent, half_dead, is_gone, move_id, last_move_id, second_last_move_id, move_base_damage, move_adjusted_damage, move_hits)
        monster.powers = [Power.from_json(json_power) for json_power in json_object["powers"]]
        return monster

    def __eq__(self, other):
        if self.name == other.name and self.current_hp == other.current_hp and self.max_hp == other.max_hp and self.block == other.block:
            if len(self.powers) == len(other.powers):
                for i in range(len(self.powers)):
                    if self.powers[i] != other.powers[i]:
                        return False
                return True
        return False
    
    def to_json(self):
        return {
            "name": self.name,
            "max_hp": self.max_hp,
            "current_hp": self.current_hp,
            "block": self.block,
            "half_dead": self.half_dead,
            "is_gone": self.is_gone,
        }

if __name__ == "__main__":
    # 测试Monster
    # monster = Monster("test", 1, 100, 100, 0, Intent.ATTACK, False, False, 0, None, None, 10, 10, 3)
    # vec = monster.get_vector()
    # print(vec)

    orb = {"passive_amount":3,"name":"闪电","id":"Lightning","evoke_amount":8}
    # 测试orb的读取
    # orb_obj = Orb.from_json(orb)
    # print(orb_obj.get_vector())

    player_json = {
        "orbs": [],
        "current_hp": 36,
        "block": 0,
        "max_hp": 70,
        "powers": [
            {
                "amount": 3,
                "just_applied": False,
                "name": "虚弱",
                "id": "Weakened"
            }
        ],
        "energy": 1
    }
    player_obj = Player.from_json(player_json)
    print(player_obj.get_vector())