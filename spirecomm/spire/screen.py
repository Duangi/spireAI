from enum import Enum
from typing import List, Tuple
import torch
import torch.nn as nn

from spirecomm.spire.potion import Potion
from spirecomm.spire.card import Card
from spirecomm.spire.relic import Relic
from spirecomm.spire.map import Map, Node
from spirecomm.utils.data_processing import get_hash_id, normal_normalize, _pad_vector_list,norm_linear_clip
from spirecomm.ai.constants import MAX_DECK_SIZE, MAX_HAND_SIZE, MAX_MAP_NODE_COUNT, MAX_SCREEN_ITEM_FEAT_DIM, MAX_SCREEN_ITEMS, MAX_SCREEN_MISC_DIM

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

    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成通用的屏幕上下文 Tensor。
        
        Returns:
            screen_type: int
            misc_feats: [SCREEN_MISC_DIM] (Float)
            item_ids:   [MAX_SCREEN_ITEMS] (Long) # 告诉模型屏幕上有什么，它们查表得到的embedding id
            item_feats: [MAX_SCREEN_ITEMS, 2] (Float) # 告诉模型屏幕上物品的特征（价格/状态等）
        """
        screen_type_val = self.SCREEN_TYPE.value
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        # 基类默认返回空，子类覆盖逻辑填充
        return screen_type_val, misc_feats, item_ids, item_feats

    def _fill_items(self, 
                    items: list, 
                    item_ids: torch.Tensor, 
                    item_feats: torch.Tensor, 
                    get_id_func, 
                    get_feat_func):
        """通用填充辅助函数"""
        count = 0
        for item in items:
            if count >= MAX_SCREEN_ITEMS: break
            
            # 填 ID
            item_ids[count] = get_id_func(item)
            # 填 特征 (价格/状态)
            feats = get_feat_func(item)
            item_feats[count] = torch.tensor(feats, dtype=torch.float32)
            
            count += 1

class ChestScreen(Screen):

    SCREEN_TYPE = ScreenType.CHEST

    def __init__(self, chest_type: ChestType, chest_open: bool):
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

    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        # 根据from_json的chest_type字段生成唯一id
        item_ids[0] = get_hash_id(self.chest_type.name + "Chest")
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        item_feats[0, 0] = 1.0 if self.chest_open else 0.0  # chest_open状态作为特征
        
        return screen_type_val, misc_feats, item_ids, item_feats

class EventScreen(Screen):

    SCREEN_TYPE = ScreenType.EVENT

    def __init__(self, name, event_id, body_text=""):
        super().__init__()
        self.event_name = name
        self.event_id = event_id
        self.body_text = body_text
        self.options: List[EventOption] = []

    @classmethod
    def from_json(cls, json_object):
        event = cls(json_object["event_name"], json_object["event_id"], json_object["body_text"])
        for json_option in json_object["options"]:
            event.options.append(EventOption.from_json(json_option))
        return event
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        【接口 2】用于 Screen Context 生成
        返回:
            screen_type: int
            misc_feats: [SCREEN_MISC_DIM] (Float)
            item_ids:   [MAX_SCREEN_ITEMS] (Long)
            item_feats: [MAX_SCREEN_ITEMS, 2] (Float)
        """
        screen_type_val = self.SCREEN_TYPE.value
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        # 这里就填一下具体是哪一个事件吧
        misc_feats[0] = get_hash_id(self.event_name)
        misc_feats[1] = get_hash_id(self.body_text)
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        # 用label作为id，因为label通常与choice_list里的选项字符串一致
        for i, option in enumerate(self.options):
            if i >= MAX_SCREEN_ITEMS:
                break
            item_ids[i] = get_hash_id(option.label if option.label else option.text)
        # 不需要也根本没有feats，知道id就行
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)

        return screen_type_val, misc_feats, item_ids, item_feats
        
        


class ShopRoomScreen(Screen):

    SCREEN_TYPE = ScreenType.SHOP_ROOM

class RestScreen(Screen):

    SCREEN_TYPE = ScreenType.REST

    def __init__(self, has_rested, rest_options: List[RestOption]):
        super().__init__()
        self.has_rested = has_rested
        self.rest_options = rest_options

    @classmethod
    def from_json(cls, json_object):
        rest_options = [RestOption[option.upper()] for option in json_object.get("rest_options")]
        return cls(json_object.get("has_rested"), rest_options)
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = 1.0 if self.has_rested else 0.0
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        for i, option in enumerate(self.rest_options):
            if i >= MAX_SCREEN_ITEMS:
                break
            item_ids[i] = get_hash_id(option.name)
        
        return screen_type_val, misc_feats, item_ids, item_feats


class CardRewardScreen(Screen):

    SCREEN_TYPE = ScreenType.CARD_REWARD

    def __init__(self, cards: List[Card], can_bowl: bool, can_skip: bool):
        super().__init__()
        self.cards = cards
        self.can_bowl = can_bowl # 应该是放弃然后回2血
        self.can_skip = can_skip

    @classmethod
    def from_json(cls, json_object):
        cards = [Card.from_json(card) for card in json_object.get("cards")]
        can_bowl = json_object.get("bowl_available")
        can_skip = json_object.get("skip_available")
        return cls(cards, can_bowl, can_skip)
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = 1.0 if self.can_bowl else 0.0
        misc_feats[1] = 1.0 if self.can_skip else 0.0
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        for i, card in enumerate(self.cards):
            if i >= MAX_SCREEN_ITEMS:
                break
            # 用name作为id，因为choice_list里出现卡牌时，是以name为准的。
            emb_id, feats = card.get_tensor_data()
            item_ids[i] = emb_id
            item_feats[i, :feats.numel()] = feats

        return screen_type_val, misc_feats, item_ids, item_feats

class CombatReward:

    def __init__(self, reward_type: RewardType, gold=0, relic=None, potion=None, link=None):
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

    def __init__(self, rewards: List[CombatReward]):
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
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        for i, reward in enumerate(self.rewards):
            if i >= MAX_SCREEN_ITEMS:
                break
            # 填特征
            if reward.reward_type in [RewardType.GOLD, RewardType.STOLEN_GOLD]:
                item_ids[i] = get_hash_id("Gold_Reward")
                item_feats[i, 0] = norm_linear_clip(reward.gold, 300)  # 假设最大300金币
            elif reward.reward_type == RewardType.RELIC and reward.relic is not None:
                # relic的counter作为特征
                emb_id, feats = reward.relic.get_tensor_data()
                item_ids[i] = emb_id
                item_feats[i, :feats.numel()] = feats
            elif reward.reward_type == RewardType.POTION and reward.potion is not None:
                # potion的potency作为特
                emb_id, feats = reward.potion.get_tensor_data()
                item_ids[i] = emb_id
                item_feats[i, :feats.numel()] = feats
            elif reward.reward_type == RewardType.SAPPHIRE_KEY and reward.link is not None:
                # 进阶20的钥匙奖励
                item_ids[i] = get_hash_id("Sapphire_Key_Reward")
        return screen_type_val, misc_feats, item_ids, item_feats

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
    
    def process_map_to_tensors(self,game_map: Map, current_x: int, current_y: int):
        """
        将 Map 对象转化为模型所需的 Tensor
        """
        if self.current_node is None:
            raise ValueError("当前节点不能为空以处理地图")
        current_x = self.current_node.x
        current_y = self.current_node.y
        # 初始化 Tensor (全部填0/Padding)
        # node_ids: LongTensor
        node_ids = torch.zeros(MAX_MAP_NODE_COUNT, dtype=torch.long)
        # coords: FloatTensor
        node_coords = torch.zeros((MAX_MAP_NODE_COUNT, 2), dtype=torch.float32)
        # mask: FloatTensor
        reachable_mask = torch.zeros(MAX_MAP_NODE_COUNT, dtype=torch.float32)
        if game_map is not None and game_map.nodes_flattened:
            # 1. 获取可达性列表 (Python List[bool])
            # 使用你刚刚修复好的 get_reachable_mask
            bool_mask = game_map.get_reachable_mask(current_x, current_y)
            
            # 2. 遍历所有节点进行填充
            # 只取前 MAX_MAP_NODE_COUNT 个，防止越界
            num_nodes = min(len(game_map.nodes_flattened), MAX_MAP_NODE_COUNT)
            
            for i in range(num_nodes):
                node = game_map.nodes_flattened[i]
                
                # A. 填入 ID (你的 type_id 属性，已包含 BOSS 逻辑)
                node_ids[i] = node.type_id 
                
                # B. 填入坐标 (你的 get_pos_features 方法)
                node_coords[i] = node.get_pos_features()
                
                # C. 填入掩码
                if bool_mask[i]:
                    reachable_mask[i] = 1.0
                else:
                    reachable_mask[i] = 0.0

        return node_ids, node_coords, reachable_mask

    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = 1.0 if self.boss_available else 0.0
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        return screen_type_val, misc_feats, item_ids, item_feats

class BossRewardScreen(Screen):

    SCREEN_TYPE = ScreenType.BOSS_REWARD

    def __init__(self, relics: List[Relic]):
        super().__init__()
        self.relics = relics

    @classmethod
    def from_json(cls, json_object):
        relics = [Relic.from_json(relic) for relic in json_object.get("relics")]
        return cls(relics)
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        for i, relic in enumerate(self.relics):
            if i >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = relic.get_tensor_data()
            item_ids[i] = emb_id
            item_feats[i, :feats.numel()] = feats

        return screen_type_val, misc_feats, item_ids, item_feats


class ShopScreen(Screen):

    SCREEN_TYPE = ScreenType.SHOP_SCREEN

    def __init__(self, cards: List[Card], relics: List[Relic], potions: List[Potion], purge_available: bool, purge_cost: int):
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
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = 1.0 if self.purge_available else 0.0
        misc_feats[1] = norm_linear_clip(self.purge_cost, 300)
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        count = 0
        # 填卡牌
        for card in self.cards:
            if count >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = card.get_tensor_data()
            item_ids[count] = emb_id
            item_feats[count, :feats.numel()] = feats
            count += 1
        # 填遗物
        for relic in self.relics:
            if count >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = relic.get_tensor_data()
            item_ids[count] = emb_id
            item_feats[count, :feats.numel()] = feats
            count += 1
        # 填药水
        for potion in self.potions:
            if count >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = potion.get_tensor_data()
            item_ids[count] = emb_id
            item_feats[count, :feats.numel()] = feats
            count += 1

        return screen_type_val, misc_feats, item_ids, item_feats
        


class GridSelectScreen(Screen):

    SCREEN_TYPE = ScreenType.GRID

    def __init__(self, cards:List[Card], selected_cards:List[Card], num_cards:int, any_number:bool, confirm_up:bool, for_upgrade:bool, for_transform:bool, for_purge:bool):
        super().__init__()
        self.cards = cards
        self.selected_cards = selected_cards
        self.num_cards = num_cards # 可选择的卡牌数量
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
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = norm_linear_clip(self.num_cards, 10) # 预见？预见100张牌？倒不是没有这个可能
        misc_feats[1] = 1.0 if self.any_number else 0.0
        misc_feats[2] = 1.0 if self.confirm_up else 0.0
        misc_feats[3] = 1.0 if self.for_upgrade else 0.0
        misc_feats[4] = 1.0 if self.for_transform else 0.0
        misc_feats[5] = 1.0 if self.for_purge else 0.0
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        # 现在扩充到了120个槽位，前100个放卡牌，后20个放已选择的卡牌
        for i, card in enumerate(self.cards):
            if i >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = card.get_tensor_data()
            item_ids[i] = emb_id
            item_feats[i, :feats.numel()] = feats
        offset = 100
        for i, card in enumerate(self.selected_cards):
            if i + offset >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = card.get_tensor_data()
            item_ids[i + offset] = emb_id
            item_feats[i + offset, :feats.numel()] = feats

        return screen_type_val, misc_feats, item_ids, item_feats

class HandSelectScreen(Screen):

    SCREEN_TYPE = ScreenType.HAND_SELECT

    def __init__(self, cards:List[Card], selected_cards:List[Card], num_cards:int, can_pick_zero:bool):
        super().__init__()
        self.cards = cards
        self.selected_cards = selected_cards
        self.num_cards = num_cards
        self.can_pick_zero = can_pick_zero

    @classmethod
    def from_json(cls, json_object):
        cards = [Card.from_json(card) for card in json_object.get("hand")]
        selected_cards = [Card.from_json(card) for card in json_object.get("selected")]
        num_cards = json_object.get("max_cards")
        can_pick_zero = json_object.get("can_pick_zero")
        return cls(cards, selected_cards, num_cards, can_pick_zero)
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = norm_linear_clip(self.num_cards, MAX_HAND_SIZE)
        misc_feats[1] = 1.0 if self.can_pick_zero else 0.0
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        # 前10个槽位放手牌，后10个槽位放已选择的牌
        for i, card in enumerate(self.cards):
            if i >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = card.get_tensor_data()
            item_ids[i] = emb_id
            item_feats[i, :feats.numel()] = feats
        offset = 10
        for i, card in enumerate(self.selected_cards):
            if i + offset >= MAX_SCREEN_ITEMS:
                break
            emb_id, feats = card.get_tensor_data()
            item_ids[i + offset] = emb_id
            item_feats[i + offset, :feats.numel()] = feats

        return screen_type_val, misc_feats, item_ids, item_feats

class GameOverScreen(Screen):

    SCREEN_TYPE = ScreenType.GAME_OVER

    def __init__(self, score, victory):
        super().__init__()
        self.score = score
        self.victory = victory

    @classmethod
    def from_json(cls, json_object):
        return cls(json_object.get("score"), json_object.get("victory"))
    
    def get_tensor_data(self) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        screen_type_val = self.SCREEN_TYPE.value
        
        misc_feats = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
        misc_feats[0] = norm_linear_clip(self.score, 10000) # 假设最高10000分
        misc_feats[1] = 1.0 if self.victory else 0.0
        
        item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
        item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)
        
        return screen_type_val, misc_feats, item_ids, item_feats


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