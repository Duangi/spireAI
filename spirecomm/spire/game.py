from enum import Enum
import xxhash

from spirecomm.spire.relic import Relic
from spirecomm.spire.character import Player, Monster, PlayerClass
from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.spire.map import Map,Node
from spirecomm.spire.screen import Screen, ScreenType, screen_from_json, RestOption
from spirecomm.ai.constants import MAX_MAP_NODE_COUNT
from spirecomm.ai.constants import MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_DECK_SIZE, MAX_POTION_COUNT

from spirecomm.utils.data_processing import _pad_vector_list

from spirecomm.utils.data_processing import minmax_normalize, get_hash_val_normalized, normal_normalize
from typing import List
import torch
import torch.nn as nn
from dataclasses import dataclass, field

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

    current_action: str # 不量化
    current_hp: int # 当前血量，采用minmax量化
    max_hp: int # 最大血量，采用正态归一化，最小值为1，p99为80
    floor: int # 层数，采用minmax，最小值1，最大值60
    act: int # 采用one-hot编码总共是1-4 四种
    gold: int
    seed: int # 不量化
    character: PlayerClass # one-hot编码，总共四种
    ascension_level: int
    relics: List[Relic]
    deck: List[Card]
    potions: List[Potion]
    map: Map
    act_boss: str

    # Combat state
    in_game: bool
    in_combat: bool
    player: Player
    monsters: List[Monster]
    draw_pile: List[Card]
    discard_pile: List[Card]
    exhaust_pile: List[Card]
    hand: List[Card]

    # Fields with default values
    limbo: List[Card] = field(default_factory=list) # 不量化
    card_in_play: Card = None # 不量化
    turn: int = 0 # 正态分布量化0-10
    cards_discarded_this_turn: int = 0 # 正态分布量化0-5

    # Screen state
    screen: Screen = None # 不用量化了
    screen_up: bool = False # 不用量化了
    screen_type: ScreenType = ScreenType.NONE # one-hot量化，ScreenType的类型数量
    room_phase: RoomPhase = None # one-hot量化，RoomPhase的类型数量
    room_type: str = None # 先跳过不量化，后续知道所有房间类别之后再回来 TODO
    choice_list: List[str] = field(default_factory=list) # 不用量化
    choice_available: bool = False # 不用量化

    available_commands: List[str] = field(default_factory=list) # 不量化

    

    def get_available_command_vector(self):
        """根据 available_commands 生成一个 multi-hot 编码的向量"""
        vec = torch.zeros(len(CommandType), dtype=torch.float32)
        for command in self.available_commands:
            if command in COMMAND_MAP:
                vec[COMMAND_MAP[command].value] = 1.0
        return vec

    def get_vector(self):
        """返回所有能够量化的指标，融合进一个vector中用作输入的向量。"""
        parts = []
        # 基本标量特征
        parts.append(torch.tensor([normal_normalize(self.max_hp, 1, 80)], dtype=torch.float32))
        parts.append(torch.tensor([minmax_normalize(self.current_hp, 0, max(1, self.max_hp))], dtype=torch.float32))
        parts.append(torch.tensor([minmax_normalize(self.floor, 1, 60)], dtype=torch.float32))
        parts.append(nn.functional.one_hot(torch.tensor(int(self.act) - 1, dtype=torch.long), num_classes=4).float())
        parts.append(torch.tensor([normal_normalize(self.gold, 0, 300)], dtype=torch.float32))

        # character one-hot (use one_hot for clarity)
        parts.append(nn.functional.one_hot(
            torch.tensor(int(self.character.value) - 1, dtype=torch.long), 
            num_classes=len(PlayerClass)).float()
        )

        # ascension
        parts.append(torch.tensor([minmax_normalize(self.ascension_level if self.ascension_level is not None else 0, 0, 20)], dtype=torch.float32))

        # relics: pad/truncate to 100 relics (use Relic.get_vector)
        relic_vec_size = Relic.get_vec_length()
        parts.append(_pad_vector_list([(r.get_vector()) for r in (self.relics or [])[:100]], 100, vec_size=relic_vec_size))

        # deck, draw, discard, exhaust: each card vector size = 20 (Card.get_vector)
        card_vec_size = Card.get_vec_length()
        parts.append(_pad_vector_list([(c.get_vector()) for c in (self.deck or [])[:MAX_DECK_SIZE]], MAX_DECK_SIZE, vec_size=card_vec_size))
        parts.append(_pad_vector_list([(c.get_vector()) for c in (self.draw_pile or [])[:MAX_DECK_SIZE]], MAX_DECK_SIZE, vec_size=card_vec_size))
        parts.append(_pad_vector_list([(c.get_vector()) for c in (self.discard_pile or [])[:MAX_DECK_SIZE]], MAX_DECK_SIZE, vec_size=card_vec_size))
        parts.append(_pad_vector_list([(c.get_vector()) for c in (self.exhaust_pile or [])[:MAX_DECK_SIZE]], MAX_DECK_SIZE, vec_size=card_vec_size))

        # hand (max 10)
        parts.append(_pad_vector_list([(c.get_vector()) for c in (self.hand or [])[:MAX_HAND_SIZE]], MAX_HAND_SIZE, vec_size=card_vec_size))

        # potions (max 5)
        potion_vec_size = Potion.get_vec_length()
        parts.append(_pad_vector_list([(p.get_vector()) for p in (self.potions or [])[:MAX_POTION_COUNT]], MAX_POTION_COUNT, vec_size=potion_vec_size))

        # map vector: pad/truncate to MAX_MAP_NODE_COUNT * 14
        map_node_size = Node.get_vec_length()
        map_vec = torch.zeros(MAX_MAP_NODE_COUNT * map_node_size)
        if self.map is not None:
            mvec = self.map.get_vector()
            mlen = mvec.numel()
            take = min(mlen, MAX_MAP_NODE_COUNT * map_node_size)
            if take > 0:
                map_vec[:take] = mvec.flatten()[:take]
        parts.append(map_vec)

        # act_boss: simple one-hot per act (3 choices per act). If unknown, zero vector
        act_boss_options = {
            1: ["Slime Boss", "The Guardian", "Hexaghost"],
            2: ["The Champ", "The Collector", "Bronze Automaton"],
            3: ["Awakened One", "Time Eater", "Donu and Deca"]
        }
        boss_onehot = torch.zeros(3, dtype=torch.float32)
        opts = act_boss_options.get(self.act, None)
        if opts and self.act_boss in opts:
            boss_onehot[opts.index(self.act_boss)] = 1.0
            # 如果是第四幕的最终boss的话，默认就是000，也可以，算是信息给到位了
        parts.append(boss_onehot)

        # combat related: in_combat, player (we encode player.class if available), monsters
        parts.append(torch.tensor([1.0 if self.in_combat else 0.0], dtype=torch.float32))
        
        # player vector
        if self.player is not None:
            parts.append(self.player.get_vector())
        else:
            # 如果不在战斗中，则用一个零向量填充
            parts.append(torch.zeros(Player.get_vec_length()))

        # Playable card mask
        playable_mask = torch.zeros(MAX_HAND_SIZE, dtype=torch.float32)
        if self.hand:
            for i, card in enumerate(self.hand[:MAX_HAND_SIZE]):
                if card.is_playable:
                    playable_mask[i] = 1.0
        parts.append(playable_mask)

        # Target monster mask
        target_mask = torch.zeros(MAX_MONSTER_COUNT, dtype=torch.float32)
        if self.monsters:
            for i, monster in enumerate(self.monsters[:MAX_MONSTER_COUNT]):
                if not monster.is_gone:
                    target_mask[i] = 1.0
        parts.append(target_mask)

        # monsters (pad to MAX_MONSTER_COUNT)
        monster_vec_size = Monster.get_vec_length()
        parts.append(_pad_vector_list([(m.get_vector()) for m in (self.monsters or [])[:MAX_MONSTER_COUNT]], MAX_MONSTER_COUNT, default_size=monster_vec_size))

        # turn and cards_discarded_this_turn
        parts.append(torch.tensor([normal_normalize(self.turn, 0, 10)], dtype=torch.float32))
        parts.append(torch.tensor([normal_normalize(self.cards_discarded_this_turn, 0, 5)], dtype=torch.float32))

        # screen_type and room_phase one-hot
        screen_type_len = len(ScreenType)
        if self.screen_type is not None:
            parts.append(nn.functional.one_hot(torch.tensor(int(self.screen_type.value) - 1, dtype=torch.long), num_classes=screen_type_len).to(torch.float32))
        else:
            parts.append(torch.zeros(screen_type_len, dtype=torch.float32))

        room_phase_len = len(RoomPhase)
        if self.room_phase is not None:
            parts.append(nn.functional.one_hot(torch.tensor(int(self.room_phase.value) - 1, dtype=torch.long), num_classes=room_phase_len).to(torch.float32))
        else:
            parts.append(torch.zeros(room_phase_len, dtype=torch.float32))

        # screen vector (now handled by Screen classes)
        parts.append(self.screen.get_vector() if self.screen else torch.zeros(Screen.get_vec_length()))

        # available commands flags
        parts.append(self.get_available_command_vector())

        result = torch.cat([p.flatten() for p in parts])
        # 将result转成xxhash摘要保存到self._vector_hash中以备后续比较
        self._vector_hash = xxhash.xxh64(result.detach().cpu().numpy().tobytes()).hexdigest()
        return result
        
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
        if game.choice_available:
            game.choice_list = json_state.get("choice_list")

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
    def __eq__(self, other):
        """严格比较：如果两个 Game 的量化向量字节完全相同则视为相等（高效）。"""
        if self == None or other == None:
            return False
        if not isinstance(other, Game):
            return False
        # 快速同对象判断
        if self is other:
            return True
        try:
            if self._vector_hash is not None and other._vector_hash is not None:
                return self._vector_hash == other._vector_hash
            else:
                return torch.equal(self.get_vector().detach().cpu(), other.get_vector().detach().cpu())
        except Exception:
            return False

if __name__ == "__main__":
    pass