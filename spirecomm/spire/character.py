from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import logging

import torch
import torch.nn as nn
from spirecomm.spire.power import Power
from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized, normal_normalize, norm_linear_clip, norm_log, get_hash_id
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
    EMPTY = 5

class Orb:

    def __init__(self, name, orb_id, evoke_amount, passive_amount):
        self.name = name
        self.orb_id = orb_id
        self.evoke_amount = evoke_amount
        self.passive_amount = passive_amount

    @classmethod
    def from_json(cls, json_object):
        name = json_object.get("name")
        orb_id = json_object.get("id")
        evoke_amount = json_object.get("evoke_amount")
        passive_amount = json_object.get("passive_amount")
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
        self.orbs: List[Orb] = []
    @classmethod
    def get_vec_length(cls):
        # block, energy, orbs, powers
        return 1 + 1 + MAX_ORB_COUNT * Orb.get_vec_length() + MAX_POWER_COUNT * Power.get_vec_length()
    def get_tensor_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        numeric_vec: [4] (HP, MaxHP, Block, Energy)
        power_ids:   [MAX_POWER_COUNT]
        power_feats:  [MAX_POWER_COUNT]
        orb_ids:     [MAX_ORB_COUNT]
        orb_vals:    [MAX_ORB_COUNT, 2] (Evoke, Passive)
        """
            
        # ==========================================
        # 1. 数值特征 (Numeric)
        # ==========================================
        # 目标：生成 [current_hp, max_hp, hp_ratio, block, energy]
        hp_ratio = self.current_hp / (self.max_hp + 1e-5)
        numeric_vec = torch.tensor([
            norm_linear_clip(self.current_hp, 100.0),
            norm_linear_clip(self.max_hp, 100.0),
            hp_ratio,
            norm_linear_clip(self.block, 50.0),
            norm_linear_clip(self.energy, 5.0)
        ], dtype=torch.float32)
        # ==========================================
        # 2. 能力特征 (Powers)
        # ==========================================
        # 目标：生成 [MAX_POWER_COUNT] 的 ID 和 特征
        power_ids = torch.zeros(MAX_POWER_COUNT, dtype=torch.long)
        power_feats = torch.zeros((MAX_POWER_COUNT, 3), dtype=torch.float32) 
        
        current_powers = self.powers if self.powers else []
        for i, p in enumerate(current_powers[:MAX_POWER_COUNT]):
            power_ids[i] = get_hash_id(p.power_id)
            power_feats[i, 0] = norm_log(p.amount, 20.0)
            power_feats[i, 1] = norm_log(p.damage, 20.0) 
            power_feats[i, 2] = 1.0 if p.just_applied else 0.0

        # ==========================================
        # 3. 充能球特征 (Orbs)
        # ==========================================
        # 目标：生成 [MAX_ORB_COUNT] 的 ID 和 数值
        orb_ids = torch.zeros(MAX_ORB_COUNT, dtype=torch.long)
        orb_vals = torch.zeros((MAX_ORB_COUNT, 2), dtype=torch.float32) # [Evoke, Passive]

        current_orbs = self.orbs if self.orbs else []
        
        # 我们遍历固定的 MAX_ORB_COUNT 长度
        for i in range(MAX_ORB_COUNT):
            if i < len(current_orbs):
                # --- 情况 A: 列表内的球（可能是真球，也可能是空槽位）---
                orb = current_orbs[i]
                
                # 安全获取属性（防止 None）
                o_name = str(orb.name) if orb.name else ""
                o_id = str(orb.orb_id) if orb.orb_id else ""
                
                # 判断逻辑：根据名字或ID判断是否为空槽位
                # 游戏通常返回 name="充能球栏位" 或 id="Empty"
                if "充能球栏位" in o_name or "Empty" in o_id:
                    # 这是一个可用的空位
                    orb_ids[i] = get_hash_id("OrbSlot_Empty")
                    # 数值保持为 0
                elif o_id:
                    # 这是一个真的球 (Lightning, Frost, Dark, Plasma...)
                    # 直接 Hash 它的 ID 字符串
                    orb_ids[i] = get_hash_id(o_id)
                    
                    # 归一化数值 (Dark球的evoke可能会很高，上限设大点)
                    orb_vals[i, 0] = norm_log(orb.evoke_amount, 60.0)
                    orb_vals[i, 1] = norm_log(orb.passive_amount, 30.0)
                else:
                    # 数据异常，视作 Padding
                    pass
            else:
                # --- 情况 B: 列表外的部分（超过当前球位上限）---
                # 比如上限3个球，i=3,4...9 就是这种情况
                # 保持 orb_ids[i] = 0 (Padding)
                # 模型会学到：ID=0 的位置是无法生成球的
                pass
        
        return numeric_vec, power_ids, power_feats, orb_ids, orb_vals
    
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
    
    def get_tensor_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回 Monster 的结构化 Tensor 数据。
        
        Returns:
            numeric_vec: [N] (HP, Block, Damage, Hits, Is_Attack...)
            identity_id: [1] (Name+ID Hash，代表“我是哪只怪”)
            intent_id:   [1] (Intent Name Hash，代表“我这回合想干嘛”)
            power_ids:   [MAX_POWERS]
            power_feats: [MAX_POWERS, 3]
        """
        
        # ---------------------------------------------------
        # 1. 身份 ID (Identity)
        # ---------------------------------------------------
        # 结合 Name 和 ID，区分同名怪
        unique_name = f"{self.name}_{self.monster_id}"
        identity_id = torch.tensor([get_hash_id(unique_name)], dtype=torch.long)

        # ---------------------------------------------------
        # 2. 意图 ID (Intent)
        # ---------------------------------------------------
        # 将 Intent 枚举的名字 (如 "ATTACK_BUFF") 映射为 ID
        # 这样模型可以通过 Embedding 理解意图的类型
        intent_str = self.intent.name if self.intent else "UNKNOWN"
        intent_id = torch.tensor([get_hash_id(intent_str)], dtype=torch.long)

        # ---------------------------------------------------
        # 3. 数值特征 (Numeric)
        # ---------------------------------------------------
        hp_ratio = self.current_hp / (self.max_hp + 1e-5)
        
        # 提取伤害数值
        # 注意：如果意图不是攻击，damage 通常为 0，这很好，反映了真实情况
        base_dmg = self.move_base_damage if self.move_base_damage else 0
        adj_dmg = self.move_adjusted_damage if self.move_adjusted_damage else 0
        hits = self.move_hits if self.move_hits else 0
        
        # 提取是否攻击 (Boolean)
        # 利用 Enum 里的辅助函数
        is_attacking = 1.0 if (self.intent and self.intent.is_attack()) else 0.0

        numeric_vec = torch.tensor([
            # 生存属性
            norm_linear_clip(self.current_hp, 300.0),
            norm_linear_clip(self.max_hp, 300.0),
            hp_ratio,
            norm_linear_clip(self.block, 80.0),
            
            # 攻击属性 (关键！)
            norm_log(adj_dmg, 60.0),      # 调整后伤害 (Log归一化)
            norm_linear_clip(hits, 15.0), # 攻击段数
            is_attacking,                 # 是否攻击标志位
            
            # 状态位
            1.0 if self.half_dead else 0.0, # 觉醒者/心脏等复活怪
            1.0 if self.is_gone else 0.0
        ], dtype=torch.float32)

        # ---------------------------------------------------
        # 4. 能力特征 (Powers)
        # ---------------------------------------------------
        power_ids = torch.zeros(MAX_POWER_COUNT, dtype=torch.long)
        power_feats = torch.zeros((MAX_POWER_COUNT, 3), dtype=torch.float32) # [层数, 伤害, 刚施加]
        
        current_powers = self.powers if self.powers else []
        for i, p in enumerate(current_powers[:MAX_POWER_COUNT]):
            power_ids[i] = get_hash_id(p.power_id)
            
            # 细分 Power 特征
            power_feats[i, 0] = norm_log(p.amount, 20.0)      # 层数
            power_feats[i, 1] = norm_log(p.damage, 20.0)      # 伤害 (如中毒/荆棘)
            power_feats[i, 2] = 1.0 if p.just_applied else 0.0 # 刚施加
            
        return numeric_vec, identity_id, intent_id, power_ids, power_feats
    
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