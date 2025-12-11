from enum import Enum
import xxhash

from spirecomm.ai.dqn_core.model import SpireState
from spirecomm.spire.relic import Relic
from spirecomm.spire.character import Player, Monster, PlayerClass
from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.spire.map import Map,Node
from spirecomm.spire.screen import Screen, ScreenType, screen_from_json, RestOption
from spirecomm.ai.constants import MAX_MAP_NODE_COUNT
from spirecomm.ai.constants import MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_DECK_SIZE, MAX_POTION_COUNT, MAX_CHOICE_LIST

from spirecomm.utils.data_processing import _pad_vector_list

from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized, normal_normalize, norm_linear_clip, norm_log, norm_ratio
from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import numpy as np
import logging

class RoomPhase(Enum):
    COMBAT = 1
    EVENT = 2
    COMPLETE = 3
    INCOMPLETE = 4

class CommandType(Enum):
    END_TURN = 0
    POTION = 1
    PLAY = 2
    PROCEED = 3
    CANCEL = 4

COMMAND_MAP = {
    "end": CommandType.END_TURN,
    # 兼容性：支持旧的 "potion" 字符串，同时接受明确的 "potion_use"/"potion_discard"
    "potion": CommandType.POTION,
    "potion_use": CommandType.POTION,
    "potion_discard": CommandType.POTION,
    "play": CommandType.PLAY,
    "proceed": CommandType.PROCEED, "confirm": CommandType.PROCEED,
    "cancel": CommandType.CANCEL, "leave": CommandType.CANCEL, "return": CommandType.CANCEL, "skip": CommandType.CANCEL
}

@dataclass
class Game:
    state_hash: int = field(init=False, default=None)
    
    # 数值向量
    current_hp: int # 当前血量 
    max_hp: int # 最大血量
    floor: int # 层数
    act: int # 采用one-hot编码总共是1-4 四种
    gold: int
    character: PlayerClass # one-hot编码，总共四种
    ascension_level: int
    act_boss: str # one-hot编码
    # Combat state
    in_game: bool
    in_combat: bool
    turn: int = 0 # 0-10
    cards_discarded_this_turn: int = 0 # 0-5
    screen_up: bool = False
    choice_available: bool = False
    screen_type: ScreenType = ScreenType.NONE # one-hot量化，ScreenType的类型数量
    available_commands: List[str] = field(default_factory=list) # 不量化，直接用multi-hot编码
    room_phase: RoomPhase = None # one-hot量化，RoomPhase的类型数量
    

    # Embedding层数据,后续需要通过SumPooling方式处理
    # TODO 如果效果不好，考虑使用Transformer等更复杂的结构
    
    deck: List[Card] = field(default_factory=list)
    draw_pile: List[Card] = field(default_factory=list)
    discard_pile: List[Card] = field(default_factory=list)
    exhaust_pile: List[Card] = field(default_factory=list)

    hand: List[Card] = field(default_factory=list)

    relics: List[Relic] = field(default_factory=list)

    potions: List[Potion] = field(default_factory=list)
    
    choice_list: List[str] = field(default_factory=list) # 非常需要量化
    
    card_in_play: Card = None # embedding层数据

    player: Player = None

    monsters: List[Monster] = field(default_factory=list)
    
    map: Map = None
    
    
    
    
    screen: Screen = None # 直接在screen类里处理量化
    

    # 这些都是不量化的
    room_type: str = None # 先跳过不量化，后续知道所有房间类别之后再回来 TODO
    current_action: str = None # 不量化
    seed: int = 0 # 不量化
    limbo: List[Card] = field(default_factory=list) # 不量化
    

    def get_available_command_vector(self) -> torch.Tensor:
        """根据 available_commands 生成一个 multi-hot 编码的向量"""
        vec = torch.zeros(10, dtype=torch.float32)
        for command in self.available_commands:
            # 排除掉 key, click, wait, state 等无实际意义的命令
            if command in ["key", "click", "wait", "state"]:
                continue
            command_index_map = {"choose": 0, "return": 1, "play": 2, "end": 3, "proceed": 4, "skip": 5, "potion": 6, "leave": 7, "confirm": 8, "cancel": 9}
            if command in command_index_map:
                vec[command_index_map[command]] = 1.0
        return vec

    def get_numeric_vector(self) -> torch.Tensor:
        """
        仅返回数值型量化特征的向量（不包含embedding等复杂结构）。
        用于快速比较游戏状态的数值差异。
        """
        parts = []
        # 基本标量特征
        # 最大血量 120就很多了
        parts.append(torch.tensor([norm_log(self.max_hp, 120)], dtype=torch.float32))
        # 当前血量
        parts.append(torch.tensor([norm_log(self.current_hp, 120)], dtype=torch.float32))
        # 当前血量比例
        parts.append(torch.tensor([norm_ratio(self.current_hp, self.max_hp)], dtype=torch.float32))
        # 层数 
        parts.append(torch.tensor([norm_linear_clip(self.floor, 60)], dtype=torch.float32)) 
        # Act one-hot
        parts.append(nn.functional.one_hot(torch.tensor(int(self.act) - 1, dtype=torch.long), num_classes=4).float())
        # Gold 1000表示很多
        parts.append(torch.tensor([norm_log(self.gold, 1000)], dtype=torch.float32))

        # 角色 one-hot (use one_hot for clarity)
        parts.append(nn.functional.one_hot(
            torch.tensor(int(self.character.value) - 1, dtype=torch.long), 
            num_classes=len(PlayerClass)).float()
        )

        # 进阶难度
        parts.append(torch.tensor([minmax_normalize(self.ascension_level if self.ascension_level is not None else 0, 0, 20)], dtype=torch.float32))

        # act_boss: simple one-hot per act (3 choices per act). If unknown, zero vector
        boss_onehot = self._get_act_boss_onehot()
        parts.append(boss_onehot)

        # in_combat
        parts.append(torch.tensor([1.0 if self.in_combat else 0.0], dtype=torch.float32))

        # turn 10回合了还没启动薄纱对面也该死了
        parts.append(torch.tensor([norm_linear_clip(self.turn, 0, 10)], dtype=torch.float32))

        # cards_discarded_this_turn 一般来说丢牌只查有没有丢过，或者丢过了减费，也就是5张就够了
        parts.append(torch.tensor([norm_linear_clip(self.cards_discarded_this_turn, 0, 5)], dtype=torch.float32))
        
        # screen_up
        parts.append(torch.tensor([1.0 if self.screen_up else 0.0], dtype=torch.float32))

        # choice_available
        parts.append(torch.tensor([1.0 if self.choice_available else 0.0], dtype=torch.float32))

        # screen_type one-hot
        parts.append(nn.functional.one_hot(
            torch.tensor(int(self.screen_type.value) - 1, dtype=torch.long), 
            num_classes=len(ScreenType)
        ).float())

        # room_phase one-hot
        parts.append(nn.functional.one_hot(
            torch.tensor(int(self.room_phase.value) - 1, dtype=torch.long), 
            num_classes=len(RoomPhase)
        ).float())

        # 读取{"choose": 0, "return": 1, "play": 2, "end": 3, "proceed": 4, "skip": 5, "potion": 6, "leave": 7, "confirm": 8, "cancel": 9}
        # available commands flags, 希望模型能够根据这个学习到当前哪些命令虽然是可用，但实际上后续是会被mask掉的
        parts.append(self.get_available_command_vector())
        
        result = torch.cat([p.flatten() for p in parts])
        return result
    def _get_act_boss_onehot(self) -> torch.Tensor:
        """返回 act_boss 的 one-hot 编码向量"""
        act_boss_options = {
            1: ["Slime Boss", "The Guardian", "Hexaghost"],
            2: ["The Champ", "The Collector", "Bronze Automaton"],
            3: ["Awakened One", "Time Eater", "Donu and Deca"]
        }
        boss_onehot = torch.zeros(3, dtype=torch.float32)
        if self.act == 4:
            # 第四幕最终boss，默认111
            boss_onehot = torch.ones(3, dtype=torch.float32)
        else:
            opts = act_boss_options.get(self.act, None)
            if opts and self.act_boss in opts:
                boss_onehot[opts.index(self.act_boss)] = 1.0
            else:
                raise ValueError(f"无法为 act {self.act} 确定 act_boss 的 one-hot 编码，act_boss={self.act_boss}")
        return boss_onehot
    def generate_vector(self) -> SpireState:
        """
        生成量化向量，用于匹配SpireDQN模型的输入格式。
        返回 SpireState 对象
        """



    @classmethod
    def from_json(cls, communication_state, available_commands=None):
        # 使用 __new__ 创建实例以避免调用需要参数的 __init__
        game = cls.__new__(cls)

        # 兼容旧的调用方式，同时处理新的顶层状态
        # 如果 available_commands 是 None，说明传入的是完整的 communication_state
        json_state = communication_state.get("game_state", {}) if available_commands is None else communication_state
        available_commands = communication_state.get("available_commands", []) if available_commands is None else available_commands
        
        game.in_game = communication_state.get("in_game", False)
        game.current_action = json_state.get("current_action", None)
        game.current_hp = json_state.get("current_hp")
        game.max_hp = json_state.get("max_hp")
        game.floor = json_state.get("floor")
        game.act = json_state.get("act")
        game.gold = json_state.get("gold")
        game.seed = json_state.get("seed")
        game.character = PlayerClass[json_state.get("class")]
        game.ascension_level = json_state.get("ascension_level")
        game.relics = [Relic.from_json(json_relic) for json_relic in json_state.get("relics")]
        game.deck = [Card.from_json(json_card) for json_card in json_state.get("deck")]
        game.map = Map.from_json(json_state.get("map"))
        game.potions = [Potion.from_json(potion) for potion in json_state.get("potions")]
        game.act_boss = json_state.get("act_boss", None)

        # Screen State

        game.screen_up = json_state.get("is_screen_up", False)
        game.screen_type = ScreenType[json_state.get("screen_type")]
        game.screen = screen_from_json(game.screen_type, json_state.get("screen_state"))
        game.room_phase = RoomPhase[json_state.get("room_phase")]
        game.room_type = json_state.get("room_type")
        game.choice_available = "choice_list" in json_state
        # Ensure choice_list is always a list, even if empty
        game.choice_list = json_state.get("choice_list", []) if game.choice_available else []

        # Combat state
        # 无论是否在战斗中，都初始化这些属性以避免AttributeError
        game.player = None
        game.monsters = []
        game.draw_pile = []
        game.discard_pile = []
        game.exhaust_pile = []
        game.hand = []
        game.limbo = []
        game.in_combat = game.room_phase == RoomPhase.COMBAT
        if game.in_combat:
            combat_state = json_state.get("combat_state")
            game.player = Player.from_json(combat_state.get("player"))
            game.monsters = [Monster.from_json(json_monster) for json_monster in combat_state.get("monsters")]
            for i, monster in enumerate(game.monsters):
                monster.monster_index = i
            game.draw_pile = [Card.from_json(json_card) for json_card in combat_state.get("draw_pile")]
            game.discard_pile = [Card.from_json(json_card) for json_card in combat_state.get("discard_pile")]
            game.exhaust_pile = [Card.from_json(json_card) for json_card in combat_state.get("exhaust_pile")]
            game.hand = [Card.from_json(json_card) for json_card in combat_state.get("hand")]
            game.limbo = [Card.from_json(json_card) for json_card in combat_state.get("limbo", [])]
            game.card_in_play = combat_state.get("card_in_play", None)
            if game.card_in_play is not None:
                game.card_in_play = Card.from_json(game.card_in_play)
            game.turn = combat_state.get("turn", 0)
            game.cards_discarded_this_turn = combat_state.get("cards_discarded_this_turn", 0)

        game.available_commands = available_commands
        # 确保任何缓存的向量摘要被清除（因为该实例刚被构建/修改）
        try:
            game._invalidate_vector_hash()
        except Exception:
            game._vector_hash = None

        return game

    def are_potions_full(self):
        for potion in self.potions:
            if potion.potion_id == "Potion Slot":
                return False
        return True
    
    def __eq__(self, value):
        if not isinstance(value, Game):
            return False
        if value is None:
            return False
        if self is value:
            return True
        if self.state_hash is not None and value.state_hash is not None:
            return self.state_hash == value.state_hash
        else:
            return False
            