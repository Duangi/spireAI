from dataclasses import dataclass, fields
import json
import torch.nn as nn
import torch.nn.functional as F
import xxhash
from spirecomm.ai.constants import MAX_HAND_SIZE, MAX_MONSTER_COUNT, MAX_DECK_SIZE, MAX_POTION_COUNT,MAX_VOCAB_SIZE,MAX_SCREEN_ITEM_FEAT_DIM, MAX_SCREEN_MISC_DIM, MAX_SCREEN_ITEMS
from spirecomm.ai.dqn_core.action import NUM_ACTION_TYPES
import torch

class DQNModel(nn.Module): # 这实际上是一个 Dueling Branching Q-Network
    def __init__(self, state_size):
        super(DQNModel, self).__init__()
        # 共享主体 (Shared Body)
        self.shared_layer1 = nn.Linear(state_size, 512)
        self.shared_layer2 = nn.Linear(512, 512)
        self.shared_layer3 = nn.Linear(512, 256)

        # --- 优势函数 (Advantage) 的“头” ---
        # 评估每个动作/参数相对于平均水平的“优势”
        self.advantage_action_type = nn.Linear(256, NUM_ACTION_TYPES)
        self.advantage_play_card = nn.Linear(256, MAX_HAND_SIZE)
        self.advantage_target_monster = nn.Linear(256, MAX_MONSTER_COUNT)
        self.advantage_choose_option = nn.Linear(256, MAX_DECK_SIZE)
        self.advantage_potion_use = nn.Linear(256, MAX_POTION_COUNT)
        self.advantage_potion_discard = nn.Linear(256, MAX_POTION_COUNT)

        # --- 状态价值函数 (Value) 的“头” ---
        # 只评估当前状态的好坏，与具体动作无关
        self.value_head = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.shared_layer1(state))
        x = F.relu(self.shared_layer2(x))
        x = F.relu(self.shared_layer3(x))

        # 计算状态价值 V(s)
        value = self.value_head(x)

        # 计算每个动作/参数的优势 A(s, a)
        adv_action_type = self.advantage_action_type(x)
        adv_play_card = self.advantage_play_card(x)
        adv_target_monster = self.advantage_target_monster(x)
        adv_choose_option = self.advantage_choose_option(x)
        # 分为两个独立的药水头（use / discard）
        adv_potion_use = self.advantage_potion_use(x)
        adv_potion_discard = self.advantage_potion_discard(x)

        # Dueling DQN 核心公式: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # 组合得到最终的Q值
        # 主动作头的Q值
        action_type_q = value + (adv_action_type - adv_action_type.mean(dim=1, keepdim=True))
        # 参数头的Q值，通常直接用优势值或加上状态价值
        play_card_q = adv_play_card
        target_monster_q = adv_target_monster
        choose_option_q = adv_choose_option
        potion_use_q = adv_potion_use
        potion_discard_q = adv_potion_discard

        # 输出类型:action_type_q: [batch_size, NUM_ACTION_TYPES]
        # 其他参数头: [batch_size, param_size]
        return action_type_q, {
            'play_card': play_card_q,
            'target_monster': target_monster_q,
            'choose_option': choose_option_q,
            'potion_use': potion_use_q,
            'potion_discard': potion_discard_q
        }

# ==========================================
# 1. 配置类 (Configuration)
# ==========================================
@dataclass
class SpireConfig:
    # --- 基础维度 (对应 Game/Character 类中的定义) ---
    numeric_global_dim: int = 17    # 全局数值特征维度
    numeric_monster_dim: int = 9    # 单个怪物的数值特征维度
    numeric_player_dim: int = 5     # 玩家数值特征维度
    
    # --- Screen 专用维度 (补全这里) ---
    # 对应 ScreenEncoder 的输入
    dim_screen_misc: int = MAX_SCREEN_MISC_DIM    # 全局杂项 (Misc Feats)
    dim_screen_item_feat: int = MAX_SCREEN_ITEM_FEAT_DIM # 物品特征 (16基础 + 1选中状态)
    max_screen_items: int = MAX_SCREEN_ITEMS  # 屏幕最大物品数
    # --- 手工特征维度 (对应各 Entity.get_tensor_data) ---
    feat_dim_card: int = 16
    feat_dim_relic: int = 3
    feat_dim_potion: int = 2
    feat_dim_power: int = 3
    feat_dim_orb: int = 2
    feat_dim_screen_item: int = MAX_SCREEN_ITEM_FEAT_DIM 
    
    # --- 词表大小 ---
    vocab_size: int = MAX_VOCAB_SIZE      # 万能字典
    room_vocab_size: int = 10   # 地图房间类型
    
    # --- 网络隐藏层维度 ---
    embed_dim: int = 128        # Embedding 标准维度
    feat_dim: int = 128         # 特征投影后的统一维度
    pooler_hidden_dim: int = 128# 列表压缩后的维度
    scorer_hidden_dim: int = 128# 打分头隐藏层
    context_dim: int = 1024     # 全局上下文维度
    
    # --- 游戏常量 ---
    num_action_types: int = 10  # 动作类型数量
    
    # --- 架构开关 ---
    use_attention_pooling: bool = False # True=使用Attention, False=使用DualPooling

# ==========================================
# 2. 输入状态契约 (Input Contract)
# ==========================================
@dataclass
class SpireState:
    # --- A. 全局数值 & 掩码 ---
    global_numeric: torch.Tensor           # [Batch, numeric_global_dim]
    action_mask: torch.Tensor              # [Batch, num_action_types]

    # --- B. 简单列表 (只查 Embedding) ---
    deck_ids: torch.Tensor                 # [Batch, 100]
    draw_pile_ids: torch.Tensor            # [Batch, 100]
    discard_pile_ids: torch.Tensor         # [Batch, 100]
    exhaust_pile_ids: torch.Tensor         # [Batch, 100]
    # limbo_ids: torch.Tensor                # [Batch, 10]
    
    # --- C. 复杂实体 (Rich Representation) ---
    # Hand
    hand_ids: torch.Tensor                 # [Batch, 10]
    hand_feats: torch.Tensor               # [Batch, 10, 16]
    
    # Relics
    relic_ids: torch.Tensor                # [Batch, 25]
    relic_feats: torch.Tensor              # [Batch, 25, 3]
    
    # Potions
    potion_ids: torch.Tensor               # [Batch, 5]
    potion_feats: torch.Tensor             # [Batch, 5, 2]
    
    # Choices
    choice_ids: torch.Tensor               # [Batch, 15]
    
    # Card In Play
    card_in_play_id: torch.Tensor          # [Batch, 1] (0 if None)
    
    # --- D. 玩家状态 ---
    player_numeric: torch.Tensor           # [Batch, numeric_player_dim]
    player_power_ids: torch.Tensor         # [Batch, 20]
    player_power_feats: torch.Tensor       # [Batch, 20, 3]
    player_orb_ids: torch.Tensor           # [Batch, 10]
    player_orb_vals: torch.Tensor          # [Batch, 10, 2]

    # --- E. 怪物状态 ---
    monster_ids: torch.Tensor              # [Batch, 5]
    monster_intent_ids: torch.Tensor       # [Batch, 5]
    monster_numeric: torch.Tensor          # [Batch, 5, numeric_monster_dim]
    monster_power_ids: torch.Tensor        # [Batch, 5, 20]
    monster_power_feats: torch.Tensor      # [Batch, 5, 20, 3]

    # --- F. 屏幕与地图 ---
    screen_type_val: torch.Tensor          # [Batch, 1]
    screen_misc: torch.Tensor              # [Batch, 8]
    screen_item_ids: torch.Tensor          # [Batch, 20]
    screen_item_feats: torch.Tensor        # [Batch, 20, 17]

    map_node_ids: torch.Tensor             # [Batch, 60]
    map_node_coords: torch.Tensor          # [Batch, 60, 2]
    map_mask: torch.Tensor                 # [Batch, 60]

    def __str__(self):
        """
        生成该状态的确定性字符串表示。
        忽略 device (CPU/GPU) 和 grad 差异，仅关注形状和数值。
        """
        state_dict = {}
        
        # 1. 自动遍历所有字段
        for field in fields(self):
            val = getattr(self, field.name)
            
            if isinstance(val, torch.Tensor):
                # 2. 标准化 Tensor
                # .detach(): 去除梯度信息
                # .cpu(): 统一移动到 CPU
                # .tolist(): 转为 Python 原生列表，消除 PyTorch 打印格式(precision/linewidth)的差异
                # 包含 shape 是为了区分数据量相同但形状不同的情况 (如 [1, 6] vs [2, 3])
                key = f"{field.name} | shape{tuple(val.shape)}"
                content = val.detach().cpu().tolist()
                state_dict[key] = content
            else:
                # 处理非 Tensor 字段 (如果有)
                state_dict[field.name] = str(val)

        # 3. 确定性序列化
        # 使用 json.dumps 配合 sort_keys=True，确保字典键的顺序永远固定
        # 这样即使字典内部存储顺序变化，生成的字符串也是一样的
        return json.dumps(state_dict, sort_keys=True, indent=None)

    # 可选：如果你需要哈希值（例如放入 set 或作为 dict 的 key）
    def __hash__(self):
        # 使用xxhash
        return xxhash.xxh64(str(self)).intdigest()


# ==========================================
# 3. 输出结果契约 (Output Contract)
# ==========================================
@dataclass
class SpireOutput:
    q_action_type: torch.Tensor    # [Batch, num_action_types]
    
    q_play_card: torch.Tensor      # [Batch, 10]
    q_target_monster: torch.Tensor # [Batch, 5]
    q_choose_option: torch.Tensor  # [Batch, 15]
    q_potion_use: torch.Tensor     # [Batch, 5]
    q_potion_discard: torch.Tensor # [Batch, 5]

# ==========================================
# 4. 聚合模块 (Pooling Strategies)
# ==========================================

class DualPooling(nn.Module):
    """
    【推荐】双路聚合：Sum + Max
    Sum 捕捉总量信息 (我有多少张打击)，Max 捕捉关键信息 (我有神级遗物)。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 输入维度翻倍，因为拼接了 sum 和 max
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, mask=None):
        """
        x: [B, Seq, Dim]
        mask: [B, Seq] (1=Valid, 0=Padding)
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            # Sum 用 0 填充
            x_sum = x * mask_expanded
            # Max 用 极小值 填充 (防止 Max 选中 0)
            x_max = x.masked_fill(mask_expanded == 0, -1e9)
        else:
            x_sum = x
            x_max = x
            
        sum_pool = x_sum.sum(dim=1)
        max_pool = x_max.max(dim=1)[0] # max 返回 (values, indices)
        
        combined = torch.cat([sum_pool, max_pool], dim=1)
        return self.proj(combined)

class AttentionPooling(nn.Module):
    """
    【备选】注意力聚合：Weighted Sum
    通过一个可学习的 Query 向量，自动寻找列表中最重要的元素。
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 全局可学习的查询向量 "我想找什么?"
        self.query = nn.Parameter(torch.randn(1, 1, input_dim))
        
        self.key_layer = nn.Linear(input_dim, input_dim)
        self.val_layer = nn.Linear(input_dim, input_dim)
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # [B, 1, Dim]
        Q = self.query.expand(batch_size, -1, -1)
        # [B, Seq, Dim]
        K = self.key_layer(x)
        V = self.val_layer(x)
        
        # Attention Scores: [B, 1, Seq]
        scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(-1) ** 0.5)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
            
        weights = torch.softmax(scores, dim=-1)
        
        # Weighted Sum: [B, 1, Dim]
        context = torch.bmm(weights, V)
        return self.proj(context.squeeze(1))

# ==========================================
# 5. 编码器组件 (Encoders)
# ==========================================

class FeatureFusion(nn.Module):
    """通用融合: Embedding + Numeric Features"""
    def __init__(self, numeric_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128 + numeric_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, emb, feats):
        return self.net(torch.cat([emb, feats], dim=2))

class PowerEncoder(nn.Module):
    """Power: ID + [Amount, Damage, JustApplied]"""
    def __init__(self, config: SpireConfig, emb_layer: nn.Embedding):
        super().__init__()
        self.emb = emb_layer
        self.feat_proc = nn.Sequential(
            nn.Linear(config.feat_dim_power, config.embed_dim),
            nn.ReLU()
        )
    def forward(self, ids, feats):
        return self.emb(ids) + self.feat_proc(feats)

class OrbEncoder(nn.Module):
    """Orb: ID + [Evoke, Passive]"""
    def __init__(self, config: SpireConfig, emb_layer: nn.Embedding):
        super().__init__()
        self.emb = emb_layer
        self.feat_proc = nn.Sequential(
            nn.Linear(config.feat_dim_orb, config.embed_dim),
            nn.ReLU()
        )
    def forward(self, ids, feats):
        return self.emb(ids) + self.feat_proc(feats)

class MapEncoder(nn.Module):
    """Map: Embed + Coords + Mask"""
    def __init__(self, config: SpireConfig):
        super().__init__()
        self.emb = nn.Embedding(config.room_vocab_size, config.embed_dim, padding_idx=0)
        self.net = nn.Sequential(
            nn.Linear(config.embed_dim + 2, config.pooler_hidden_dim),
            nn.ReLU()
        )
    def forward(self, ids, coords, mask):
        x = self.emb(ids)
        x = torch.cat([x, coords], dim=2)
        x = self.net(x)
        # Masked Sum Pooling
        return (x * mask.unsqueeze(-1)).sum(dim=1)

class ScreenEncoder(nn.Module):
    """Screen: Type + Misc + Rich Items"""
    def __init__(self, config: SpireConfig, emb_layer: nn.Embedding):
        super().__init__()
        self.emb = emb_layer
        
        # 1. 内部组件维度
        dim_type = 32
        dim_misc = 32
        dim_items = config.pooler_hidden_dim # 128
        
        self.type_emb = nn.Embedding(20, dim_type, padding_idx=0)
        self.misc_enc = nn.Linear(8, dim_misc) # Misc维度设为8
        
        self.item_feat_enc = nn.Sequential(
            nn.Linear(config.feat_dim_screen_item, config.embed_dim),
            nn.ReLU()
        )
        self.pooler = nn.Sequential(
            nn.Linear(config.embed_dim, dim_items),
            nn.ReLU()
        )

        # 2. 【关键修复】输出投影层
        # 将内部拼接后的维度 (32 + 32 + 128 = 192) 统一映射回 config.feat_dim (128)
        self.output_proj = nn.Sequential(
            nn.Linear(dim_type + dim_misc + dim_items, config.feat_dim),
            nn.ReLU()
        )

    def forward(self, type_id, misc, item_ids, item_feats):
        type_vec = self.type_emb(type_id).squeeze(1) # [B, 32]
        
        # misc: [B, 8] -> [B, 32]
        misc_vec = self.misc_enc(misc)
        
        # items: [B, 20, 128]
        item_sem = self.emb(item_ids)
        item_sts = self.item_feat_enc(item_feats)
        # sum pooling -> [B, 128]
        items_sum = self.pooler((item_sem + item_sts).sum(dim=1))
        
        # 拼接: [B, 192]
        concat_vec = torch.cat([type_vec, misc_vec, items_sum], dim=1)
        
        # 投影: [B, 192] -> [B, 128]
        return self.output_proj(concat_vec)
    
class ItemScorer(nn.Module):
    """Dynamic Branching Head"""
    def __init__(self, config: SpireConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.context_dim + config.feat_dim, config.scorer_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.scorer_hidden_dim, 1)
        )
    def forward(self, context, items, mask):
        ctx_exp = context.unsqueeze(1).expand(-1, items.shape[1], -1)
        combined = torch.cat([ctx_exp, items], dim=2)
        scores = self.net(combined).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        return scores

# ==========================================
# 6. 主模型 (The Brain)
# ==========================================
class SpireDQN(nn.Module):
    def __init__(self, config: SpireConfig):
        super().__init__()
        self.cfg = config
        
        # --- 1. 基础组件 --- 用生成的5000词万能字典转为128维Embedding
        self.unified_emb = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        
        # 选择聚合策略
        if config.use_attention_pooling:
            self.pooler = AttentionPooling(config.feat_dim, config.feat_dim)
        else:
            self.pooler = DualPooling(config.embed_dim, config.feat_dim)
            
        # 简单Sum Pooling (用于Deck/Discard)
        self.simple_pooler = lambda x: x.sum(dim=1)

        # --- 2. 专用编码器 ---
        self.power_enc = PowerEncoder(config, self.unified_emb)
        self.orb_enc = OrbEncoder(config, self.unified_emb)
        self.map_enc = MapEncoder(config)
        self.screen_enc = ScreenEncoder(config, self.unified_emb)
        
        # --- 3. 特征融合器 (Rich Features) ---
        self.hand_fusion = FeatureFusion(config.feat_dim_card, config.feat_dim)
        self.relic_fusion = FeatureFusion(config.feat_dim_relic, config.feat_dim)
        self.potion_fusion = FeatureFusion(config.feat_dim_potion, config.feat_dim)
        self.monster_num_proj = nn.Linear(config.numeric_monster_dim, config.embed_dim)
        
        # --- 4. 全局数值处理 ---
        self.global_num_enc = nn.Linear(config.numeric_global_dim, config.feat_dim)
        self.player_num_enc = nn.Linear(config.numeric_player_dim, config.feat_dim)

        # --- 5. Context Body ---
        # 计算 Context 输入维度 (根据你的模块数量调整)
        # Global(1) + Player(3) + Hand(1) + Deck(1) + Discard(1) + Relic(1) + 
        # Potion(1) + Monster(1) + Map(1) + Screen(1) + CardPlay(1)
        # 加上 Draw Pile 和 Exhaust Pile (各1) = 15

        total_ctx = config.feat_dim * 15
        
        self.shared_body = nn.Sequential(
            nn.Linear(total_ctx, config.context_dim),
            nn.LayerNorm(config.context_dim),
            nn.ReLU(),
            nn.Linear(config.context_dim, config.context_dim),
            nn.ReLU()
        )

        # --- 6. Heads ---
        self.value_head = nn.Linear(config.context_dim, 1)
        self.action_type_head = nn.Linear(config.context_dim, config.num_action_types)
        
        self.scorers = nn.ModuleDict({
            'card': ItemScorer(config),
            'target': ItemScorer(config),
            'choice': ItemScorer(config),
            'potion_use': ItemScorer(config),
            'potion_discard': ItemScorer(config)
        })

    def forward(self, state: SpireState) -> SpireOutput:
        # ================= Phase 1: Encoding =================
        
        # A. 简单列表
        ctx_deck = self.simple_pooler(self.unified_emb(state.deck_ids))
        ctx_discard = self.simple_pooler(self.unified_emb(state.discard_pile_ids))
        ctx_draw = self.simple_pooler(self.unified_emb(state.draw_pile_ids))
        ctx_exhaust = self.simple_pooler(self.unified_emb(state.exhaust_pile_ids))

        # B. 复杂实体 (Rich Rep + Pooling)
        # Hand
        hand_emb = self.unified_emb(state.hand_ids)
        hand_rich = self.hand_fusion(hand_emb, state.hand_feats)
        ctx_hand = self.pooler(hand_rich, mask=(state.hand_ids != 0))
        
        # Relic
        relic_emb = self.unified_emb(state.relic_ids)
        relic_rich = self.relic_fusion(relic_emb, state.relic_feats)
        ctx_relic = self.pooler(relic_rich, mask=(state.relic_ids != 0))
        
        # Potion (数量少，直接Sum)
        potion_emb = self.unified_emb(state.potion_ids)
        potion_rich = self.potion_fusion(potion_emb, state.potion_feats)
        ctx_potion = potion_rich.sum(dim=1)
        
        # C. 玩家状态
        ctx_p_num = self.player_num_enc(state.player_numeric)
        
        p_pwr_vecs = self.power_enc(state.player_power_ids, state.player_power_feats)
        ctx_p_pwr = self.pooler(p_pwr_vecs, mask=(state.player_power_ids != 0))
        
        p_orb_vecs = self.orb_enc(state.player_orb_ids, state.player_orb_vals)
        ctx_p_orb = p_orb_vecs.sum(dim=1) # Orb顺序敏感，Sum仅作为baseline
        
        # D. 怪物状态
        m_id = self.unified_emb(state.monster_ids)
        m_int = self.unified_emb(state.monster_intent_ids)
        m_num = self.monster_num_proj(state.monster_numeric)
        
        m_pwr_detail = self.power_enc(state.monster_power_ids, state.monster_power_feats)
        m_pwr_sum = m_pwr_detail.sum(dim=2) # 聚合每个怪的Power
        
        # 融合怪物特征 (Rich Vector 用于 Scorer)
        rich_monsters = m_id + m_int + m_num + m_pwr_sum
        # 聚合全局怪物特征 (Context 用)
        ctx_monster = self.pooler(rich_monsters, mask=(state.monster_ids != 0))
        
        # E. 环境与全局
        ctx_map = self.map_enc(state.map_node_ids, state.map_node_coords, state.map_mask)
        ctx_screen = self.screen_enc(state.screen_type_val, state.screen_misc, state.screen_item_ids, state.screen_item_feats)
        ctx_global = self.global_num_enc(state.global_numeric)
        ctx_card_play = self.unified_emb(state.card_in_play_id).squeeze(1)

        # ================= Phase 2: Context =================
        
        raw_context = torch.cat([
            ctx_global,
            ctx_p_num, ctx_p_pwr, ctx_p_orb,
            ctx_hand, ctx_deck, ctx_discard,
            ctx_relic, ctx_potion, ctx_monster,
            ctx_map, ctx_screen, ctx_card_play,

            # 之前漏掉了 draw和exhaust pile
            ctx_draw, ctx_exhaust
        ], dim=1)
        
        context = self.shared_body(raw_context)

        # ================= Phase 3: Scoring =================
        
        # Action Type
        val = self.value_head(context)
        adv = self.action_type_head(context)
        q_action = val + (adv - adv.mean(dim=1, keepdim=True))
        
        if state.action_mask is not None:
            q_action = q_action.masked_fill(state.action_mask == 0, -1e9)

        # Branches
        def score(name, items, ids):
            return self.scorers[name](context, items, mask=(ids != 0))

        return SpireOutput(
            q_action_type = q_action,
            q_play_card = score('card', hand_rich, state.hand_ids),
            q_target_monster = score('target', rich_monsters, state.monster_ids),
            q_choose_option = score('choice', self.unified_emb(state.choice_ids), state.choice_ids),
            q_potion_use = score('potion_use', potion_rich, state.potion_ids),
            q_potion_discard = score('potion_discard', potion_rich, state.potion_ids)
        )