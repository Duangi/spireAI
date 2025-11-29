from enum import Enum
import torch
import torch.nn as nn

from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.spire.relic import Relic
from spirecomm.spire.map import Node
from spirecomm.utils.data_processing import get_hash_val_normalized, normal_normalize, _pad_vector_list
from spirecomm.ai.constants import MAX_DECK_SIZE, MAX_HAND_SIZE

class ScreenType(Enum):
    EVENT = 1
    CHEST = 2
    SHOP_ROOM = 3
    REST = 4
    CARD_REWARD = 5
    COMBAT_REWARD = 6
    MAP = 7
    BOSS_REWARD = 8
    SHOP_SCREEN = 9
    GRID = 10
    HAND_SELECT = 11
    GAME_OVER = 12
    COMPLETE = 13
    NONE = 14


class ChestType(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    BOSS = 4
    UNKNOWN = 5


class RewardType(Enum):
    CARD = 1
    GOLD = 2
    RELIC = 3
    POTION = 4
    STOLEN_GOLD = 5
    EMERALD_KEY = 6
    SAPPHIRE_KEY = 7


class RestOption(Enum):
    DIG = 1
    LIFT = 2
    RECALL = 3
    REST = 4
    SMITH = 5
    TOKE = 6


class EventOption:

    def __init__(self, text, label, disabled=False, choice_index=None):
        self.text = text
        self.label = label
        self.disabled = disabled
        self.choice_index = choice_index

    @classmethod
    def from_json(cls, json_object):
        text = json_object.get("text")
        label = json_object.get("label")
        disabled = json_object.get("disabled")
        choice_index = json_object.get("choice_index", None)
        return cls(text, label, disabled, choice_index)


class Screen:

    SCREEN_TYPE = ScreenType.NONE

    def __init__(self):
        self.screen_type = type(self).SCREEN_TYPE

    @classmethod
    def from_json(cls, json_object):
        return cls()

    @classmethod
    def get_vec_length(cls):
        """计算所有屏幕状态向量中的最大可能长度"""
        shop_len = 7 * Card.get_vec_length() + 3 * Relic.get_vec_length() + 3 * Potion.get_vec_length() + 2
        card_reward_len = 3 * Card.get_vec_length() + 1
        grid_len = MAX_DECK_SIZE * Card.get_vec_length() + 4
        hand_select_len = MAX_HAND_SIZE * Card.get_vec_length() + 2
        rest_len = len(RestOption)
        event_len = 1 + 5 # event_id_hash + 5 options hash
        combat_reward_len = 5 * max(Card.get_vec_length(), Relic.get_vec_length(), Potion.get_vec_length(), 1) # 5 rewards, take max size
        boss_reward_len = 3 * Relic.get_vec_length()

        return max(shop_len, card_reward_len, grid_len, hand_select_len, rest_len, event_len, combat_reward_len, boss_reward_len)

    def get_vector(self):
        """
        为当前屏幕状态创建向量。
        基类返回一个零向量。子类应覆盖此方法。
        """
        return torch.zeros(self.get_vec_length())

    def _create_padded_vector(self, content_vector: torch.Tensor):
        padded_vec = torch.zeros(self.get_vec_length())
        padded_vec[:content_vector.numel()] = content_vector
        return padded_vec

class ChestScreen(Screen):

    SCREEN_TYPE = ScreenType.CHEST

    def __init__(self, chest_type, chest_open):
        super().__init__()
        self.chest_type = chest_type
        self.chest_open = chest_open

    @classmethod
    def from_json(cls, json_object):
        java_chest_class_name = json_object.get("chest_type")
        if java_chest_class_name == "SmallChest":
            chest_type = ChestType.SMALL
        elif java_chest_class_name == "MediumChest":
            chest_type = ChestType.MEDIUM
        elif java_chest_class_name == "LargeChest":
            chest_type = ChestType.LARGE
        elif java_chest_class_name == "BossChest":
            chest_type = ChestType.BOSS
        else:
            chest_type = ChestType.UNKNOWN
        chest_open = json_object.get("chest_open")
        return cls(chest_type, chest_open)


class EventScreen(Screen):

    SCREEN_TYPE = ScreenType.EVENT

    def __init__(self, name, event_id, body_text=""):
        super().__init__()
        self.event_name = name
        self.event_id = event_id
        self.body_text = body_text
        self.options = []

    @classmethod
    def from_json(cls, json_object):
        event = cls(json_object["event_name"], json_object["event_id"], json_object["body_text"])
        for json_option in json_object["options"]:
            event.options.append(EventOption.from_json(json_option))
        return event
    
    def get_vector(self):
        event_id_hash = get_hash_val_normalized(self.event_id)
        option_hashes = torch.zeros(5)
        for i, option in enumerate(self.options[:5]):
            option_hashes[i] = get_hash_val_normalized(option.text)
        content_vec = torch.cat([event_id_hash, option_hashes])
        return self._create_padded_vector(content_vec)


class ShopRoomScreen(Screen):

    SCREEN_TYPE = ScreenType.SHOP_ROOM


class RestScreen(Screen):

    SCREEN_TYPE = ScreenType.REST

    def __init__(self, has_rested, rest_options):
        super().__init__()
        self.has_rested = has_rested
        self.rest_options = rest_options

    @classmethod
    def from_json(cls, json_object):
        rest_options = [RestOption[option.upper()] for option in json_object.get("rest_options")]
        return cls(json_object.get("has_rested"), rest_options)
    
    def get_vector(self):
        options_one_hot = torch.zeros(len(RestOption))
        for option in self.rest_options:
            options_one_hot[option.value - 1] = 1.0
        return self._create_padded_vector(options_one_hot)


class CardRewardScreen(Screen):

    SCREEN_TYPE = ScreenType.CARD_REWARD

    def __init__(self, cards, can_bowl, can_skip):
        super().__init__()
        self.cards = cards
        self.can_bowl = can_bowl
        self.can_skip = can_skip

    @classmethod
    def from_json(cls, json_object):
        cards = [Card.from_json(card) for card in json_object.get("cards")]
        can_bowl = json_object.get("bowl_available")
        can_skip = json_object.get("skip_available")
        return cls(cards, can_bowl, can_skip)
    
    def get_vector(self):
        card_vectors = [card.get_vector() for card in self.cards[:3]]
        padded_cards = _pad_vector_list(card_vectors, 3, Card.get_vec_length())
        skip_vec = torch.tensor([1.0 if self.can_skip else 0.0], dtype=torch.float32)
        return self._create_padded_vector(torch.cat([padded_cards, skip_vec]))


class CombatReward:

    def __init__(self, reward_type, gold=0, relic=None, potion=None, link=None):
        self.reward_type = reward_type
        self.gold = gold
        self.relic = relic
        self.potion = potion
        self.link = link

    def __eq__(self, other):
        return self.reward_type == other.reward_type and self.gold == other.gold \
               and self.relic == other.relic and self.potion == other.potion and self.link == other.link


class CombatRewardScreen(Screen):

    SCREEN_TYPE = ScreenType.COMBAT_REWARD

    def __init__(self, rewards):
        super().__init__()
        self.rewards = rewards

    @classmethod
    def from_json(cls, json_object):
        rewards = []
        for json_reward in json_object.get("rewards"):
            reward_type = RewardType[json_reward.get("reward_type")]
            if reward_type in [RewardType.GOLD, RewardType.STOLEN_GOLD]:
                rewards.append(CombatReward(reward_type, gold=json_reward.get("gold")))
            elif reward_type == RewardType.RELIC:
                rewards.append(CombatReward(reward_type, relic=Relic.from_json(json_reward.get("relic"))))
            elif reward_type == RewardType.POTION:
                rewards.append(CombatReward(reward_type, potion=Potion.from_json(json_reward.get("potion"))))
            elif reward_type == RewardType.SAPPHIRE_KEY:
                rewards.append(CombatReward(reward_type, link=Relic.from_json(json_reward.get("link"))))
            else:
                rewards.append(CombatReward(reward_type))
        return cls(rewards)
    
    def get_vector(self):
        reward_vectors = []
        # We need a unified size for different reward types
        # Let's use the max size of card, relic, or potion vector
        max_size = max(Card.get_vec_length(), Relic.get_vec_length(), Potion.get_vec_length(), 1)
        for reward in self.rewards[:5]:
            vec = torch.zeros(max_size)
            if reward.reward_type == RewardType.GOLD:
                vec[0] = normal_normalize(reward.gold, 10, 100)
            elif reward.reward_type == RewardType.RELIC and reward.relic:
                r_vec = reward.relic.get_vector(); vec[:r_vec.numel()] = r_vec
            elif reward.reward_type == RewardType.POTION and reward.potion:
                p_vec = reward.potion.get_vector(); vec[:p_vec.numel()] = p_vec
            elif reward.reward_type == RewardType.CARD:
                vec[0] = -1.0 # Special value to indicate a card reward choice
            reward_vectors.append(vec)
        return self._create_padded_vector(_pad_vector_list(reward_vectors, 5, max_size))


class MapScreen(Screen):

    SCREEN_TYPE = ScreenType.MAP

    def __init__(self, current_node, next_nodes, boss_available):
        super().__init__()
        self.current_node = current_node
        self.next_nodes = next_nodes
        self.boss_available = boss_available

    @classmethod
    def from_json(cls, json_object):
        current_node_json = json_object.get("current_node", None)
        next_nodes_json = json_object.get("next_nodes", None)
        boss_available = json_object.get("boss_available")
        if current_node_json is not None:
            current_node = Node.from_json(current_node_json)
        else:
            current_node = None
        if next_nodes_json is not None:
            next_nodes = [Node.from_json(node) for node in next_nodes_json]
        else:
            next_nodes = []
        return cls(current_node, next_nodes, boss_available)


class BossRewardScreen(Screen):

    SCREEN_TYPE = ScreenType.BOSS_REWARD

    def __init__(self, relics):
        super().__init__()
        self.relics = relics

    @classmethod
    def from_json(cls, json_object):
        relics = [Relic.from_json(relic) for relic in json_object.get("relics")]
        return cls(relics)
    
    def get_vector(self):
        relic_vectors = [relic.get_vector() for relic in self.relics[:3]]
        padded_relics = _pad_vector_list(relic_vectors, 3, Relic.get_vec_length())
        return self._create_padded_vector(padded_relics)


class ShopScreen(Screen):

    SCREEN_TYPE = ScreenType.SHOP_SCREEN

    def __init__(self, cards, relics, potions, purge_available, purge_cost):
        super().__init__()
        self.cards = cards
        self.relics = relics
        self.potions = potions
        self.purge_available = purge_available
        self.purge_cost = purge_cost

    @classmethod
    def from_json(cls, json_object):
        cards = [Card.from_json(card) for card in json_object.get("cards")]
        relics = [Relic.from_json(relic) for relic in json_object.get("relics")]
        potions = [Potion.from_json(potion) for potion in json_object.get("potions")]
        purge_available = json_object.get("purge_available")
        purge_cost = json_object.get("purge_cost")
        return cls(cards, relics, potions, purge_available, purge_cost)
    
    def get_vector(self):
        card_vectors = [card.get_vector() for card in self.cards[:7]]
        padded_cards = _pad_vector_list(card_vectors, 7, Card.get_vec_length())

        relic_vectors = [relic.get_vector() for relic in self.relics[:3]]
        padded_relics = _pad_vector_list(relic_vectors, 3, Relic.get_vec_length())

        potion_vectors = [potion.get_vector() for potion in self.potions[:3]]
        padded_potions = _pad_vector_list(potion_vectors, 3, Potion.get_vec_length())

        purge_vec = torch.tensor([1.0 if self.purge_available else 0.0, normal_normalize(self.purge_cost, 50, 150)], dtype=torch.float32)

        content_vec = torch.cat([padded_cards, padded_relics, padded_potions, purge_vec])
        return self._create_padded_vector(content_vec)


class GridSelectScreen(Screen):

    SCREEN_TYPE = ScreenType.GRID

    def __init__(self, cards, selected_cards, num_cards, any_number, confirm_up, for_upgrade, for_transform, for_purge):
        super().__init__()
        self.cards = cards
        self.selected_cards = selected_cards
        self.num_cards = num_cards
        self.any_number = any_number
        self.confirm_up = confirm_up
        self.for_upgrade = for_upgrade
        self.for_transform = for_transform
        self.for_purge = for_purge

    @classmethod
    def from_json(cls, json_object):
        cards = [Card.from_json(card) for card in json_object.get("cards")]
        selected_cards = [Card.from_json(card) for card in json_object.get("selected_cards")]
        num_cards = json_object.get("num_cards")
        any_number = json_object.get("any_number", False)
        confirm_up = json_object.get("confirm_up")
        for_upgrade = json_object.get("for_upgrade")
        for_transform = json_object.get("for_transform")
        for_purge = json_object.get("for_purge")
        return cls(cards, selected_cards, num_cards, any_number, confirm_up, for_upgrade, for_transform, for_purge)
    
    def get_vector(self):
        purpose_vec = torch.tensor([
            1.0 if self.for_upgrade else 0.0,
            1.0 if self.for_transform else 0.0,
            1.0 if self.for_purge else 0.0,
            normal_normalize(self.num_cards, 1, 10)
        ], dtype=torch.float32)
        card_vectors = [card.get_vector() for card in self.cards[:MAX_DECK_SIZE]]
        return self._create_padded_vector(torch.cat([purpose_vec, _pad_vector_list(card_vectors, MAX_DECK_SIZE, Card.get_vec_length())]))


class HandSelectScreen(Screen):

    SCREEN_TYPE = ScreenType.HAND_SELECT

    def __init__(self, cards, selected, num_cards, can_pick_zero):
        super().__init__()
        self.cards = cards
        self.selected_cards = selected
        self.num_cards = num_cards
        self.can_pick_zero = can_pick_zero

    @classmethod
    def from_json(cls, json_object):
        cards = [Card.from_json(card) for card in json_object.get("hand")]
        selected_cards = [Card.from_json(card) for card in json_object.get("selected")]
        num_cards = json_object.get("max_cards")
        can_pick_zero = json_object.get("can_pick_zero")
        return cls(cards, selected_cards, num_cards, can_pick_zero)
    
    def get_vector(self):
        info_vec = torch.tensor([
            normal_normalize(self.num_cards, 1, 10),
            1.0 if self.can_pick_zero else 0.0
        ], dtype=torch.float32)
        card_vectors = [card.get_vector() for card in self.cards[:MAX_HAND_SIZE]]
        return self._create_padded_vector(torch.cat([info_vec, _pad_vector_list(card_vectors, MAX_HAND_SIZE, Card.get_vec_length())]))


class GameOverScreen(Screen):

    SCREEN_TYPE = ScreenType.GAME_OVER

    def __init__(self, score, victory):
        super().__init__()
        self.score = score
        self.victory = victory

    @classmethod
    def from_json(cls, json_object):
        return cls(json_object.get("score"), json_object.get("victory"))
    
    def get_vector(self):
        game_over_vec = torch.tensor([
            1.0 if self.victory else 0.0,
            normal_normalize(self.score, 0, 2000)
        ], dtype=torch.float32)
        return self._create_padded_vector(game_over_vec)


class CompleteScreen(Screen):

    SCREEN_TYPE = ScreenType.COMPLETE


SCREEN_CLASSES = {
    ScreenType.EVENT: EventScreen,
    ScreenType.CHEST: ChestScreen,
    ScreenType.SHOP_ROOM: ShopRoomScreen,
    ScreenType.REST: RestScreen,
    ScreenType.CARD_REWARD: CardRewardScreen,
    ScreenType.COMBAT_REWARD: CombatRewardScreen,
    ScreenType.MAP: MapScreen,
    ScreenType.BOSS_REWARD: BossRewardScreen,
    ScreenType.SHOP_SCREEN: ShopScreen,
    ScreenType.GRID: GridSelectScreen,
    ScreenType.HAND_SELECT: HandSelectScreen,
    ScreenType.GAME_OVER: GameOverScreen,
    ScreenType.COMPLETE: CompleteScreen,
    ScreenType.NONE: Screen
}


def screen_from_json(screen_type, json_object):
    return SCREEN_CLASSES[screen_type].from_json(json_object)

if __name__ == "__main__":
    import json
    # 避免循环导入，仅在测试时导入
    from spirecomm.spire.game import Game

    # 辅助函数，用于打印和断言
    def assert_and_print(name, tensor, expected_len, full_preview=False):
        actual_len = tensor.numel()
        assert actual_len == expected_len, f"{name} 长度错误! 预期: {expected_len}, 实际: {actual_len}"
        print(f"\n[ {name} ] - 长度: {actual_len}")
        if full_preview:
            preview = tensor.flatten()
        else:
            preview = tensor.flatten()[:10]
        print(f"  - 内容预览: {preview.tolist()}")

    # --- 测试 CardRewardScreen ---
    print("\n\n--- 测试 CardRewardScreen 的向量化 ---")
    card_reward_state_json = """
    {"available_commands":["choose","skip","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["杂技","灾祸","带毒刺击"],"screen_type":"CARD_REWARD","screen_state":{"cards":[{"exhausts":false,"is_playable":false,"cost":1,"name":"杂技","id":"Acrobatics","type":"SKILL","ethereal":false,"uuid":"a1fdfc68-cca6-4e48-97ae-7ce9ee8b2218","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"is_playable":false,"cost":1,"name":"灾祸","id":"Bane","type":"ATTACK","ethereal":false,"uuid":"4b90e05c-0d41-4fdf-81a1-9ff9de75f8ef","upgrades":0,"rarity":"COMMON","has_target":true},{"exhausts":false,"is_playable":false,"cost":1,"name":"带毒刺击","id":"Poisoned Stab","type":"ATTACK","ethereal":false,"uuid":"75dcd791-470a-494b-b6dc-e7f63c948a69","upgrades":0,"rarity":"COMMON","has_target":true}],"bowl_available":false,"skip_available":true},"seed":-164215452355420946,"deck":[],"relics":[],"max_hp":70,"act_boss":"The Guardian","gold":99,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"CARD_REWARD","room_phase":"COMPLETE","is_screen_up":true,"potions":[],"current_hp":70,"floor":0,"ascension_level":0,"class":"THE_SILENT","map":[]}}
    """
    game_json_reward = json.loads(card_reward_state_json)
    available_commands_reward = game_json_reward.get("available_commands", [])
    game_state_data_reward = game_json_reward.get("game_state", {})
    game_obj_reward = Game.from_json(game_state_data_reward, available_commands_reward)

    # 提取 screen vector
    screen_vector = game_obj_reward.screen.get_vector()
    
    # 手动构建 screen vector 以供对比
    manual_parts = []
    
    card_vectors = [card.get_vector() for card in game_obj_reward.screen.cards[:3]]
    padded_cards = _pad_vector_list(card_vectors, 3, Card.get_vec_length())
    manual_parts.append(padded_cards)

    skip_vec = torch.tensor([1.0 if game_obj_reward.screen.can_skip else 0.0], dtype=torch.float32)
    manual_parts.append(skip_vec)

    content_vec = torch.cat(manual_parts)
    manual_screen_vector = torch.zeros(Screen.get_vec_length())
    manual_screen_vector[:content_vec.numel()] = content_vec
    
    assert torch.equal(screen_vector, manual_screen_vector), "CardRewardScreen 的向量内容不匹配!"
    print("✅ CardRewardScreen 的向量化逻辑验证成功！")
