from dataclasses import dataclass, field

from sympy import false
from spirecomm.ai.constants import (
    MAX_CHOOSE_COUNT, MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_POTION_COUNT, 
    MAX_DECK_SIZE, MAX_ORB_COUNT, MAX_POWER_COUNT, MAX_MAP_NODE_COUNT, 
    MAX_SCREEN_ITEMS, MAX_SCREEN_MISC_DIM, MAX_SCREEN_ITEM_FEAT_DIM, MAX_CHOICE_LIST
)
from spirecomm.spire import relic
from spirecomm.spire.game import Game
from spirecomm.ai.dqn_core.action import  BaseAction, PlayAction, ChooseAction, PotionDiscardAction, PotionUseAction, SingleAction, ActionType, DecomposedActionType
from typing import List
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from spirecomm.ai.dqn_core.model import SpireState
from spirecomm.utils.data_processing import get_hash_id, minmax_normalize, norm_linear_clip, norm_log, norm_ratio
from spirecomm.spire.card import Card, CardType, CardRarity
from spirecomm.spire.relic import Relic
from spirecomm.spire.potion import Potion
from spirecomm.spire.character import Player, Monster, PlayerClass
from spirecomm.spire.map import Map, Node
from spirecomm.spire.screen import CombatReward, Screen, ScreenType, RewardType

@dataclass
class GameStateProcessor:
    """
    游戏状态预处理器。
    它的主要职责是将 `Game` 对象转换为一个扁平化的 PyTorch 张量 (tensor)，
    以便作为神经网络的输入。
    这个实现将具体的向量化逻辑委托给 `Game` 对象自身的 `get_vector` 方法。
    """
    shop_visited: bool = False
    cards_visited: List[bool] = None
    current_card_reward_index: int = -1 # 标记当前进入的是第几个卡牌奖励房间，方便记录访问情况
    max_card_reward_count: int = 0

    def get_state_tensor(self, game: Game) -> SpireState:
        """
        从 Game 对象获取完整的 SpireState 对象。
        返回的 SpireState 中各 Tensor 均为非 Batch 维度 (即 Batch=1 的情况需自行 unsqueeze 或 collate)。
        """
        
        # ==========================================
        # 1. Global Numeric (18 dim)
        # ==========================================
        # MaxHP(1) + CurHP(1) + Ratio(1) + Floor(1) + Act(4) + Gold(1) + Class(4) + Ascension(1) + Boss(3) + isOverkillBlock(1)
        
        # Act One-hot
        act_onehot = F.one_hot(torch.tensor(int(game.act) - 1, dtype=torch.long).clamp(0, 3), num_classes=4).float()
        
        # Class One-hot
        class_onehot = F.one_hot(torch.tensor(int(game.character.value) - 1, dtype=torch.long).clamp(0, 3), num_classes=4).float()
        
        # Boss One-hot
        act_boss_options = {
            1: ["Slime Boss", "The Guardian", "Hexaghost"],
            2: ["The Champ", "The Collector", "Bronze Automaton"],
            3: ["Awakened One", "Time Eater", "Donu and Deca"]
        }
        boss_onehot = torch.zeros(3, dtype=torch.float32)
        opts = act_boss_options.get(game.act, None)
        if opts and game.act_boss in opts:
            boss_onehot[opts.index(game.act_boss)] = 1.0
        # 计算所有怪物的总攻击意图值（仅当需要时使用）
        total_monster_attack_intent = 0
        for m in game.monsters:
            if m.move_adjusted_damage > 0:
                total_monster_attack_intent += m.move_adjusted_damage * m.move_hits
        is_over_kill_block = False
        if game.in_combat and game.player:
            if game.player.block > total_monster_attack_intent:
                is_over_kill_block = True
        global_numeric = torch.cat([
            torch.tensor([norm_linear_clip(game.max_hp, 80)], dtype=torch.float32),
            torch.tensor([norm_linear_clip(game.current_hp, 150)], dtype=torch.float32),
            torch.tensor([norm_ratio(game.current_hp, game.max_hp)], dtype=torch.float32),
            torch.tensor([norm_linear_clip(game.floor, 60)], dtype=torch.float32),
            act_onehot,
            torch.tensor([norm_linear_clip(game.gold, 1000)], dtype=torch.float32), # 超过1000的金币其实都没什么必要了也没什么可能了
            class_onehot,
            torch.tensor([minmax_normalize(game.ascension_level if game.ascension_level is not None else 0, 0, 20)], dtype=torch.float32),
            boss_onehot,
            torch.tensor([1.0 if is_over_kill_block else 0.0], dtype=torch.float32)
        ])

        # ==========================================
        # 2. Action Mask
        # ==========================================
        masks = self.get_action_masks(game)
        action_mask = torch.from_numpy(masks['action_type']).bool()

        # ==========================================
        # 3. Simple Lists (IDs only)
        # ==========================================
        def get_card_ids(cards, max_len):
            ids = torch.zeros(max_len, dtype=torch.long)
            for i, c in enumerate(cards[:max_len]):
                ids[i] = get_hash_id(c.name)
            return ids

        deck_ids = get_card_ids(game.deck, MAX_DECK_SIZE)
        draw_pile_ids = get_card_ids(game.draw_pile, MAX_DECK_SIZE)
        discard_pile_ids = get_card_ids(game.discard_pile, MAX_DECK_SIZE)
        exhaust_pile_ids = get_card_ids(game.exhaust_pile, MAX_DECK_SIZE)

        # ==========================================
        # 4. Hand (IDs + Feats)
        # ==========================================
        hand_ids = torch.zeros(MAX_HAND_SIZE, dtype=torch.long)
        # 假设 Card.get_tensor_data 返回 (id, feat_vec)
        # feat_vec 维度需确认，假设为 16
        hand_feats = torch.zeros((MAX_HAND_SIZE, 16), dtype=torch.float32)
        
        for i, card in enumerate(game.hand[:MAX_HAND_SIZE]):
            c_id, c_feat = card.get_tensor_data()
            hand_ids[i] = c_id
            hand_feats[i] = c_feat

        # ==========================================
        # 5. Relics
        # ==========================================
        # 假设 MAX_RELIC_COUNT = 25 (需确认常量，这里暂用25)
        MAX_RELIC_COUNT = 25 
        relic_ids = torch.zeros(MAX_RELIC_COUNT, dtype=torch.long)
        relic_feats = torch.zeros((MAX_RELIC_COUNT, 3), dtype=torch.float32)
        
        for i, relic in enumerate(game.relics[:MAX_RELIC_COUNT]):
            r_id, r_feat = relic.get_tensor_data()
            relic_ids[i] = r_id
            relic_feats[i] = r_feat

        # ==========================================
        # 6. Potions
        # ==========================================
        potion_ids = torch.zeros(MAX_POTION_COUNT, dtype=torch.long)
        potion_feats = torch.zeros((MAX_POTION_COUNT, 2), dtype=torch.float32)
        
        for i, potion in enumerate(game.potions[:MAX_POTION_COUNT]):
            p_id, p_feat = potion.get_tensor_data()
            potion_ids[i] = p_id
            # Potion.get_tensor_data 返回 4 维，截取前 2 维 (CanUse, CanDiscard)
            # 或者根据 SpireState 定义调整。这里截取前2维。
            potion_feats[i] = p_feat[:2]

        # ==========================================
        # 7. Choices
        # ==========================================
        choice_ids = torch.zeros(MAX_CHOICE_LIST, dtype=torch.long)
        choice_list = getattr(game, 'choice_list', [])
        if choice_list:
            for i, choice_str in enumerate(choice_list[:MAX_CHOICE_LIST]):
                choice_ids[i] = get_hash_id(choice_str)

        # ==========================================
        # 8. Card In Play
        # ==========================================
        card_in_play_id = torch.zeros(1, dtype=torch.long)
        if game.card_in_play:
            card_in_play_id[0] = get_hash_id(game.card_in_play.name)

        # ==========================================
        # 9. Player
        # ==========================================
        # Player.get_tensor_data returns (numeric, power_ids, power_feats, orb_ids, orb_vals)
        if game.player:
            p_num, p_pow_ids, p_pow_feats, p_orb_ids, p_orb_vals = game.player.get_tensor_data()
            
            player_numeric = p_num
            player_power_ids = p_pow_ids
            player_power_feats = p_pow_feats
            player_orb_ids = p_orb_ids
            player_orb_vals = p_orb_vals
        else:
            player_numeric = torch.zeros(5, dtype=torch.float32)
            player_power_ids = torch.zeros(MAX_POWER_COUNT, dtype=torch.long)
            player_power_feats = torch.zeros((MAX_POWER_COUNT, 3), dtype=torch.float32)
            player_orb_ids = torch.zeros(MAX_ORB_COUNT, dtype=torch.long)
            player_orb_vals = torch.zeros((MAX_ORB_COUNT, 2), dtype=torch.float32)

        # ==========================================
        # 10. Monsters
        # ==========================================
        monster_ids = torch.zeros(MAX_MONSTER_COUNT, dtype=torch.long)
        monster_intent_ids = torch.zeros(MAX_MONSTER_COUNT, dtype=torch.long)
        monster_numeric = torch.zeros((MAX_MONSTER_COUNT, 9), dtype=torch.float32) # 9 dim from Monster.get_tensor_data
        monster_power_ids = torch.zeros((MAX_MONSTER_COUNT, MAX_POWER_COUNT), dtype=torch.long)
        monster_power_feats = torch.zeros((MAX_MONSTER_COUNT, MAX_POWER_COUNT, 3), dtype=torch.float32)

        for i, monster in enumerate(game.monsters[:MAX_MONSTER_COUNT]):
            # Monster.get_tensor_data returns (numeric, identity_id, intent_id, power_ids, power_feats)
            m_num, m_id, m_intent, m_pow_ids, m_pow_feats = monster.get_tensor_data()
            
            monster_ids[i] = m_id
            monster_intent_ids[i] = m_intent
            monster_numeric[i] = m_num
            monster_power_ids[i] = m_pow_ids
            monster_power_feats[i] = m_pow_feats

        # ==========================================
        # 11. Screen
        # ==========================================
        # Screen.get_tensor_data returns (type_val, misc_feats, item_ids, item_feats)
        if game.screen:
            s_type, s_misc, s_item_ids, s_item_feats = game.screen.get_tensor_data()
            screen_type_val = torch.tensor([s_type], dtype=torch.long)
            screen_misc = s_misc
            screen_item_ids = s_item_ids
            screen_item_feats = s_item_feats
        else:
            screen_type_val = torch.zeros(1, dtype=torch.long)
            screen_misc = torch.zeros(MAX_SCREEN_MISC_DIM, dtype=torch.float32)
            screen_item_ids = torch.zeros(MAX_SCREEN_ITEMS, dtype=torch.long)
            screen_item_feats = torch.zeros((MAX_SCREEN_ITEMS, MAX_SCREEN_ITEM_FEAT_DIM), dtype=torch.float32)

        # ==========================================
        # 12. Map
        # ==========================================
        map_node_ids = torch.zeros(MAX_MAP_NODE_COUNT, dtype=torch.long)
        map_node_coords = torch.zeros((MAX_MAP_NODE_COUNT, 2), dtype=torch.float32)
        map_mask = torch.zeros(MAX_MAP_NODE_COUNT, dtype=torch.float32)

        if game.map and game.map.nodes_flattened:
            for i, node in enumerate(game.map.nodes_flattened[:MAX_MAP_NODE_COUNT]):
                map_node_ids[i] = node.type_id
                map_node_coords[i] = node.get_pos_features()
                map_mask[i] = 1.0

        result = SpireState(
            global_numeric=global_numeric,
            action_mask=action_mask,
            deck_ids=deck_ids,
            draw_pile_ids=draw_pile_ids,
            discard_pile_ids=discard_pile_ids,
            exhaust_pile_ids=exhaust_pile_ids,
            hand_ids=hand_ids,
            hand_feats=hand_feats,
            relic_ids=relic_ids,
            relic_feats=relic_feats,
            potion_ids=potion_ids,
            potion_feats=potion_feats,
            choice_ids=choice_ids,
            card_in_play_id=card_in_play_id,
            player_numeric=player_numeric,
            player_power_ids=player_power_ids,
            player_power_feats=player_power_feats,
            player_orb_ids=player_orb_ids,
            player_orb_vals=player_orb_vals,
            monster_ids=monster_ids,
            monster_intent_ids=monster_intent_ids,
            monster_numeric=monster_numeric,
            monster_power_ids=monster_power_ids,
            monster_power_feats=monster_power_feats,
            screen_type_val=screen_type_val,
            screen_misc=screen_misc,
            screen_item_ids=screen_item_ids,
            screen_item_feats=screen_item_feats,
            map_node_ids=map_node_ids,
            map_node_coords=map_node_coords,
            map_mask=map_mask
        )
        game.state_hash = hash(result)
        return result
    

    def process(self, game: Game):
        """将原始 game_state 转换为 SpireState 对象 (不增加 Batch 维度)"""
        return self.get_state_tensor(game)


    def get_action_masks(self, game_state: Game):
        """
        根据当前游戏状态，生成所有分解动作的合法性掩码。
        """
        available_actions = self.get_available_actions(game_state)
        ava_commands = game_state.available_commands
        # 移除"key","click","wait","state"等无实际意义的命令
        ava_commands = [cmd for cmd in ava_commands if cmd not in ["key", "click", "wait", "state"]]

        # 初始化所有掩码为 False
        action_type_mask = np.zeros(len(DecomposedActionType), dtype=bool)
        play_card_mask = np.zeros(MAX_HAND_SIZE, dtype=bool)
        target_monster_mask = np.zeros(MAX_MONSTER_COUNT, dtype=bool)
        choose_option_mask = np.zeros(MAX_CHOOSE_COUNT, dtype=bool)
        # 将药水掩码拆分为使用（use）和丢弃（discard）两路
        potion_use_mask = np.zeros(MAX_POTION_COUNT, dtype=bool)
        potion_discard_mask = np.zeros(MAX_POTION_COUNT, dtype=bool)

        for action in available_actions:
            # 有对应的 DecomposedActionType 才设置掩码
            if hasattr(action, 'decomposed_type'):
                decomposed_type:DecomposedActionType = action.decomposed_type
                type_val = int(decomposed_type.value)
                action_type_mask[type_val] = True

            if isinstance(action, PlayAction):
                play_card_mask[action.hand_idx] = True
                if action.target_idx is not None:
                    target_monster_mask[action.target_idx] = True
            elif isinstance(action, ChooseAction):
                # Check for potion reward when slots are full
                is_valid_choice = True
                if game_state.screen_type == ScreenType.COMBAT_REWARD and hasattr(game_state.screen, 'rewards'):
                     # Ensure index is within bounds
                     if action.choice_idx < len(game_state.screen.rewards):
                         reward = game_state.screen.rewards[action.choice_idx]
                         if reward.reward_type == RewardType.POTION and game_state.are_potions_full():
                             is_valid_choice = False
                
                if is_valid_choice:
                    choose_option_mask[action.choice_idx] = True
            elif isinstance(action, PotionUseAction):
                # 明确标记可 use 的药水位
                potion_use_mask[action.potion_idx] = True
                if action.target_idx is not None:
                    target_monster_mask[action.target_idx] = True
            elif isinstance(action, PotionDiscardAction):
                # 明确标记可 discard 的药水位
                potion_discard_mask[action.potion_idx] = True
                
        # 返回时提供独立的 potion_use / potion_discard，并保留向后兼容的 'potion'（合并）
        return {
            'action_type': action_type_mask,
            'play_card': play_card_mask,
            'target_monster': target_monster_mask,
            'choose_option': choose_option_mask,
            'potion_use': potion_use_mask,
            'potion_discard': potion_discard_mask,
            'potion': (potion_use_mask | potion_discard_mask)
        }

    def get_available_actions(self, game: Game) -> List[BaseAction]:
        """从 game_state 解析出所有合法的结构化动作对象列表"""
        actions = []
        # 以下为特殊处理：重置actions列表
        # 当房间为COMBAT_REWARD时，直接根据奖励内容，直接要求模型必须把金币和药水先选了！这是常识，避免模型需要学很多step
        if game.screen_type == ScreenType.COMBAT_REWARD:
            for i, reward in enumerate(game.screen.rewards):
                # 如果有金币，或者是药水且药水栏未满,或者遗物，或者被偷的金币
                if reward.reward_type == RewardType.GOLD or (reward.reward_type == RewardType.POTION and not game.are_potions_full()) or reward.reward_type == RewardType.RELIC or reward.reward_type == RewardType.STOLEN_GOLD:
                    # 直接只添加这些必须先选的选项
                    actions = []
                    actions.append(ChooseAction(type=ActionType.CHOOSE, choice_idx=i, decomposed_type=DecomposedActionType.CHOOSE))
                    break
        if len(actions) > 0:
            return actions
        
        # choose 动作
        if game.choice_available and "choose" in game.available_commands:
            for i, choice in enumerate(game.choice_list):
                # 访问过商店后不能再选 shop 了
                if self.shop_visited and choice == "shop":  
                    continue
                if game.screen_type == ScreenType.GRID:
                    # Grid 选择时，不能选择已经被选过的格子
                    if len(game.screen.selected_cards) > 0:
                        # 由于选择很有可能是相同的卡牌，需通过 uuid 来判断
                        selected_cards_uuids = [c.uuid for c in game.screen.selected_cards]
                        if game.screen.cards[i].uuid in selected_cards_uuids:
                            continue
                # 访问过卡牌奖励并skip之后，不能再选已经选过的卡牌了
                if game.screen_type == ScreenType.COMBAT_REWARD:
                    self.max_card_reward_count = len(game.screen.rewards)
                    if self.cards_visited:
                        if self.cards_visited[i]:
                            continue
                # 药水相关：
                if game.are_potions_full():
                    # 这里是假设有除了combat_reward以外的choose选项会有potion
                    if choice == "potion":
                        continue
                    if game.screen_type == ScreenType.COMBAT_REWARD:
                        reward = game.screen.rewards[i]
                        if reward.reward_type == RewardType.POTION:
                            continue
                    if game.screen_type == ScreenType.SHOP_SCREEN:
                        # 商店里买药水时，药水栏满了也不能买
                        potion_names = set()
                        for potion in game.screen.potions:
                            potion_names.add(potion.name)
                        if choice in potion_names:
                            continue
                # 如果有添水这个遗物，即使药水栏没满也不能买药水
                if game.screen_type == ScreenType.SHOP_SCREEN:
                    relic_names = set()
                    for relic in game.relics:
                        relic_names.add(relic.name)
                    if '添水' in relic_names:
                        potion_names = [potion.name for potion in game.screen.potions]
                        if choice in potion_names:
                            continue
                    
                actions.append(ChooseAction(type=ActionType.CHOOSE, choice_idx=i, decomposed_type=DecomposedActionType.CHOOSE))

        # 战斗中的动作
        if "play" in game.available_commands:
            # Play a card
            for hand_idx, card in enumerate(game.hand):
                if card.is_playable:
                    if card.has_target:
                        # 怪物列表直接在 game 对象下
                        for monster_idx, monster in enumerate(game.monsters):
                            if not monster.is_gone:
                                # 卡牌索引是它在手牌中的位置
                                actions.append(PlayAction(type=ActionType.PLAY, hand_idx=hand_idx, target_idx=monster_idx, decomposed_type=DecomposedActionType.PLAY))
                    else:
                        actions.append(PlayAction(type=ActionType.PLAY, hand_idx=hand_idx, target_idx=None, decomposed_type=DecomposedActionType.PLAY))
        
        # End turn
        if "end" in game.available_commands:
            actions.append(SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END))

        if "potion" in game.available_commands:
            for potion_idx, potion in enumerate(game.potions):
                if potion.can_use:
                    # 避免同时使用多个卡牌相关的药水导致卡页面
                    if game.in_combat and game.screen_type == ScreenType.CARD_REWARD:
                        # 在游戏中出现的卡牌奖励页面时，不能使用药水，先把牌选了再说！
                        continue
                    # 只允许在战斗中使用混沌药水
                    if potion.name == "混沌药水" and not game.in_combat:
                        continue
                    if potion.requires_target:
                        for monster_idx, monster in enumerate(game.monsters):
                            if not monster.is_gone:
                                # 药水索引是它在药水栏中的位置
                                actions.append(PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=monster_idx, decomposed_type=DecomposedActionType.POTION_USE))
                    else:
                        actions.append(PotionUseAction(type=ActionType.POTION_USE, potion_idx=potion_idx, target_idx=None, decomposed_type=DecomposedActionType.POTION_USE))
                if potion.can_discard:
                    # 加入masking限制扔药水！（和其他的逻辑不同，只在几种情况下允许扔药水）
                    # 情况1：腾位置：如果有战斗奖励要选药水且药水栏满了，就允许扔药水
                    if game.screen_type == ScreenType.COMBAT_REWARD:
                        if not game.are_potions_full():
                            # 如果没满，禁止扔药水
                            continue
                        else:
                            # 如果满了，且奖励里有药水，就允许扔药水腾位置
                            if game.choice_available and "choose" in game.available_commands:
                                if "potion" in game.choice_list:
                                    actions.append(PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx, decomposed_type=DecomposedActionType.POTION_DISCARD))
                    
                    # 情况2：商店里买药水时，如果药水栏满了，就允许扔药水
                    if game.screen_type == ScreenType.SHOP_SCREEN:
                        if not game.are_potions_full():
                            # 如果没满，禁止扔药水
                            continue
                        else:
                            # 如果满了，且奖励里没有药水，也不允许扔药水
                            if len(game.screen.potions) <= 0:
                                continue
                            else:
                                actions.append(PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx, decomposed_type=DecomposedActionType.POTION_DISCARD))
                    # 情况3：战斗中，有混沌药水，允许扔除了混沌药水以外的药水
                    # 情况4：战斗中，手牌中有名为炼制药水的卡牌，且手里药水已经满了，允许扔除了混沌药水以外的药水
                    if game.in_combat:
                        can_discard = False
                        if potion.name != "混沌药水":
                            if "混沌药水" in [p.name for p in game.potions]:
                                can_discard = True
                        if game.are_potions_full():
                            for card in game.hand:
                                if card.name == "炼制药水":
                                    can_discard = True
                        # 总而言之，混沌药水不能被扔
                        if potion.name == "混沌药水":
                            can_discard = False
                        if not can_discard:
                            continue
                        else:
                            actions.append(PotionDiscardAction(type=ActionType.POTION_DISCARD, potion_idx=potion_idx, decomposed_type=DecomposedActionType.POTION_DISCARD))
                    
        # 非战斗中的通用动作
        # 正确的判断方式是检查 available_commands
        if "confirm" in game.available_commands:
            actions.append(SingleAction(type=ActionType.CONFIRM, decomposed_type=DecomposedActionType.CONFIRM))
        # if "return" in game.available_commands:
        #     actions.append(SingleAction(type=ActionType.RETURN, decomposed_type=DecomposedActionType.RETURN))
        if "proceed" in game.available_commands:
            actions.append(SingleAction(type=ActionType.PROCEED, decomposed_type=DecomposedActionType.PROCEED))
        if "skip" in game.available_commands:
            actions.append(SingleAction(type=ActionType.SKIP, decomposed_type=DecomposedActionType.SKIP))
        if "leave" in game.available_commands:
            actions.append(SingleAction(type=ActionType.LEAVE, decomposed_type=DecomposedActionType.LEAVE))
        
        
        return actions
    
if __name__ == "__main__":
    from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases
    case = """{"available_commands":["choose","leave","key","click","wait","state"],"ready_for_command":true,"in_game":true,"game_state":{"choice_list":["purge","灼热攻击","飞剑回旋镖","耸肩无视","硬撑","金属化","疯狂","乱战","弹珠袋","锚","冻结之眼","铁之心","技能药水","先古药水"],"screen_type":"SHOP_SCREEN","screen_state":{"cards":[{"exhausts":false,"cost":2,"price":74,"name":"灼热攻击","id":"Searing Blow","type":"ATTACK","ethereal":false,"uuid":"78568f59-3785-4923-a4ee-a82860ae9539","upgrades":0,"rarity":"UNCOMMON","has_target":true},{"exhausts":false,"cost":1,"price":45,"name":"飞剑回旋镖","id":"Sword Boomerang","type":"ATTACK","ethereal":false,"uuid":"f43437ac-8714-4334-9eaf-4924202c0987","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"cost":1,"price":47,"name":"耸肩无视","id":"Shrug It Off","type":"SKILL","ethereal":false,"uuid":"d77590f1-0332-45fb-b6af-9b8e843d71eb","upgrades":0,"rarity":"COMMON","has_target":false},{"exhausts":false,"cost":1,"price":76,"name":"硬撑","id":"Power Through","type":"SKILL","ethereal":false,"uuid":"e90d8c55-220a-4130-ae5f-12cd131a4b49","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"cost":1,"price":34,"name":"金属化","id":"Metallicize","type":"POWER","ethereal":false,"uuid":"8c00a291-f070-453c-9d81-a216ace1e0eb","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":true,"cost":1,"price":95,"name":"疯狂","id":"Madness","type":"SKILL","ethereal":false,"uuid":"6ac1d05e-94c0-403b-8698-bc0314e103c0","upgrades":0,"rarity":"UNCOMMON","has_target":false},{"exhausts":false,"cost":2,"price":194,"name":"乱战","id":"Mayhem","type":"POWER","ethereal":false,"uuid":"c60d8037-fea4-4a4c-8348-5e518d3d56f5","upgrades":0,"rarity":"RARE","has_target":false}],"purge_cost":75,"potions":[{"requires_target":false,"can_use":false,"can_discard":true,"price":103,"name":"铁之心","id":"HeartOfIron"},{"requires_target":false,"can_use":false,"can_discard":true,"price":50,"name":"技能药水","id":"SkillPotion"},{"requires_target":false,"can_use":false,"can_discard":true,"price":76,"name":"先古药水","id":"Ancient Potion"}],"relics":[{"price":147,"name":"弹珠袋","id":"Bag of Marbles","counter":-1},{"price":144,"name":"锚","id":"Anchor","counter":-1},{"price":148,"name":"冻结之眼","id":"Frozen Eye","counter":-1}],"purge_available":true},"seed":3156255670183229424,"deck":[{"exhausts":false,"cost":1,"name":"打击","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"4349c0c8-1639-47fa-9aaf-473b69febc2c","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"打击","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"d23d02c7-752f-4d71-b2b6-bea6bd160f17","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"打击","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"c26c7784-758f-4da3-b230-ec1b65660c5b","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"打击","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"7ca525a6-b2a5-476e-ae64-27183a0563d6","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"打击","id":"Strike_R","type":"ATTACK","ethereal":false,"uuid":"a1a02ed7-ee28-4b81-9df3-ffc5473e7ecc","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":1,"name":"防御","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"0f1ab94c-b18c-42f6-8e74-e0f54fa14bb9","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"防御","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"0b9d482b-f9a2-4b5a-8ff2-559107bf9005","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"防御","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"c066475b-dff8-459f-8d19-dc25d882f7cc","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":1,"name":"防御","id":"Defend_R","type":"SKILL","ethereal":false,"uuid":"d8c9d83a-f027-4c94-a743-5765bc738ada","upgrades":0,"rarity":"BASIC","has_target":false},{"exhausts":false,"cost":2,"name":"痛击","id":"Bash","type":"ATTACK","ethereal":false,"uuid":"ccab8e24-c54a-4cc8-8f84-bdc42cfbdfe0","upgrades":0,"rarity":"BASIC","has_target":true},{"exhausts":false,"cost":-2,"name":"悔恨","id":"Regret","type":"CURSE","ethereal":false,"uuid":"b36fffc8-faae-4112-b4aa-9f6f09208750","upgrades":0,"rarity":"CURSE","has_target":false}],"relics":[{"name":"添水","id":"Sozu","counter":-1},{"name":"梨子","id":"Pear","counter":-1}],"max_hp":90,"act_boss":"Slime Boss","gold":439,"action_phase":"WAITING_ON_USER","act":1,"screen_name":"SHOP","room_phase":"COMPLETE","is_screen_up":true,"potions":[{"requires_target":false,"can_use":false,"can_discard":false,"name":"药水栏","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"药水栏","id":"Potion Slot"},{"requires_target":false,"can_use":false,"can_discard":false,"name":"药水栏","id":"Potion Slot"}],"current_hp":55,"floor":10,"ascension_level":0,"class":"IRONCLAD","map":[{"symbol":"M","children":[{"x":0,"y":1}],"x":0,"y":0,"parents":[]},{"symbol":"M","children":[{"x":4,"y":1}],"x":3,"y":0,"parents":[]},{"symbol":"M","children":[{"x":5,"y":1},{"x":6,"y":1}],"x":6,"y":0,"parents":[]},{"symbol":"?","children":[{"x":0,"y":2}],"x":0,"y":1,"parents":[]},{"symbol":"?","children":[{"x":4,"y":2}],"x":4,"y":1,"parents":[]},{"symbol":"?","children":[{"x":4,"y":2}],"x":5,"y":1,"parents":[]},{"symbol":"M","children":[{"x":6,"y":2}],"x":6,"y":1,"parents":[]},{"symbol":"M","children":[{"x":0,"y":3},{"x":1,"y":3}],"x":0,"y":2,"parents":[]},{"symbol":"M","children":[{"x":4,"y":3},{"x":5,"y":3}],"x":4,"y":2,"parents":[]},{"symbol":"M","children":[{"x":6,"y":3}],"x":6,"y":2,"parents":[]},{"symbol":"M","children":[{"x":0,"y":4}],"x":0,"y":3,"parents":[]},{"symbol":"?","children":[{"x":2,"y":4}],"x":1,"y":3,"parents":[]},{"symbol":"M","children":[{"x":3,"y":4}],"x":4,"y":3,"parents":[]},{"symbol":"?","children":[{"x":4,"y":4},{"x":6,"y":4}],"x":5,"y":3,"parents":[]},{"symbol":"M","children":[{"x":6,"y":4}],"x":6,"y":3,"parents":[]},{"symbol":"M","children":[{"x":1,"y":5}],"x":0,"y":4,"parents":[]},{"symbol":"M","children":[{"x":3,"y":5}],"x":2,"y":4,"parents":[]},{"symbol":"M","children":[{"x":3,"y":5}],"x":3,"y":4,"parents":[]},{"symbol":"?","children":[{"x":4,"y":5}],"x":4,"y":4,"parents":[]},{"symbol":"M","children":[{"x":6,"y":5}],"x":6,"y":4,"parents":[]},{"symbol":"R","children":[{"x":1,"y":6}],"x":1,"y":5,"parents":[]},{"symbol":"R","children":[{"x":2,"y":6},{"x":3,"y":6}],"x":3,"y":5,"parents":[]},{"symbol":"E","children":[{"x":5,"y":6}],"x":4,"y":5,"parents":[]},{"symbol":"R","children":[{"x":6,"y":6}],"x":6,"y":5,"parents":[]},{"symbol":"?","children":[{"x":2,"y":7}],"x":1,"y":6,"parents":[]},{"symbol":"M","children":[{"x":2,"y":7}],"x":2,"y":6,"parents":[]},{"symbol":"E","children":[{"x":3,"y":7}],"x":3,"y":6,"parents":[]},{"symbol":"R","children":[{"x":5,"y":7}],"x":5,"y":6,"parents":[]},{"symbol":"E","children":[{"x":5,"y":7},{"x":6,"y":7}],"x":6,"y":6,"parents":[]},{"symbol":"R","children":[{"x":2,"y":8},{"x":3,"y":8}],"x":2,"y":7,"parents":[]},{"symbol":"M","children":[{"x":3,"y":8}],"x":3,"y":7,"parents":[]},{"symbol":"$","children":[{"x":4,"y":8},{"x":5,"y":8}],"x":5,"y":7,"parents":[]},{"symbol":"?","children":[{"x":5,"y":8}],"x":6,"y":7,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9}],"x":2,"y":8,"parents":[]},{"symbol":"T","children":[{"x":3,"y":9},{"x":4,"y":9}],"x":3,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9}],"x":4,"y":8,"parents":[]},{"symbol":"T","children":[{"x":5,"y":9},{"x":6,"y":9}],"x":5,"y":8,"parents":[]},{"symbol":"$","children":[{"x":3,"y":10},{"x":4,"y":10}],"x":3,"y":9,"parents":[]},{"symbol":"M","children":[{"x":5,"y":10}],"x":4,"y":9,"parents":[]},{"symbol":"M","children":[{"x":5,"y":10},{"x":6,"y":10}],"x":5,"y":9,"parents":[]},{"symbol":"R","children":[{"x":6,"y":10}],"x":6,"y":9,"parents":[]},{"symbol":"M","children":[{"x":3,"y":11}],"x":3,"y":10,"parents":[]},{"symbol":"?","children":[{"x":4,"y":11}],"x":4,"y":10,"parents":[]},{"symbol":"?","children":[{"x":4,"y":11},{"x":6,"y":11}],"x":5,"y":10,"parents":[]},{"symbol":"M","children":[{"x":6,"y":11}],"x":6,"y":10,"parents":[]},{"symbol":"M","children":[{"x":2,"y":12}],"x":3,"y":11,"parents":[]},{"symbol":"M","children":[{"x":4,"y":12}],"x":4,"y":11,"parents":[]},{"symbol":"R","children":[{"x":5,"y":12},{"x":6,"y":12}],"x":6,"y":11,"parents":[]},{"symbol":"M","children":[{"x":3,"y":13}],"x":2,"y":12,"parents":[]},{"symbol":"?","children":[{"x":3,"y":13},{"x":4,"y":13}],"x":4,"y":12,"parents":[]},{"symbol":"$","children":[{"x":5,"y":13}],"x":5,"y":12,"parents":[]},{"symbol":"M","children":[{"x":5,"y":13}],"x":6,"y":12,"parents":[]},{"symbol":"M","children":[{"x":2,"y":14},{"x":4,"y":14}],"x":3,"y":13,"parents":[]},{"symbol":"E","children":[{"x":5,"y":14}],"x":4,"y":13,"parents":[]},{"symbol":"?","children":[{"x":5,"y":14},{"x":6,"y":14}],"x":5,"y":13,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":2,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":4,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":5,"y":14,"parents":[]},{"symbol":"R","children":[{"x":3,"y":16}],"x":6,"y":14,"parents":[]}],"room_type":"ShopRoom"}}
    """
    import json
    case = json.loads(case)
    # 创建两个不同的 processor 并使用独立的 game 副本
    # Wrapper fix
    if 'game_state' not in case:
        case_wrapper = {
            'game_state': case,
            'available_commands': case.get('available_commands', []),
            'in_game': case.get('in_game', True)
        }
        game1 = Game.from_json(case_wrapper)
    else:
        game1 = Game.from_json(case)

    processor1 = GameStateProcessor()
    state1 = processor1.get_state_tensor(game1)

    if game1.choice_available and "choose" in game1.available_commands:
        print("Available choices:")
        for i, choice in enumerate(game1.choice_list):
            print(f"  Choice {i}: {choice}")
            if game1.screen_type == ScreenType.SHOP_SCREEN:
                print("  Shop screen detected.")
                relic_names = set()
                for relic in game1.relics:
                    relic_names.add(relic.name)
                print(f"  Player relics: {relic_names}")
                if '添水' in relic_names:
                    potion_names = [potion.name for potion in game1.screen.potions]
                    if choice in potion_names:
                        print("Skip shop potion choice due to Sozu relic.")