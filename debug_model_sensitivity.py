import torch
import os
import sys

# 引入项目路径
sys.path.append(os.getcwd())

from spirecomm.ai.dqn_core.algorithm import SpireAgent
from spirecomm.ai.dqn_core.model import SpireConfig, SpireState
from spirecomm.utils.path import get_root_dir

def create_dummy_state(config, energy_val):
    """
    制造一个全 0 的假数据，唯独把能量设置成指定值。
    注意：这里直接生成带 Batch 维度 [1, ...] 的数据，模拟 collate 后的结果。
    """
    
    # 1. 构造 Global Numeric
    # 假设: [CurHP, MaxHP, Ratio, Floor, Act(4), Gold, Class(4), Asc, Boss(3), Overkill]
    # 我们假设 Energy 是第 4 位 (索引 4)。如果不确定，可以把所有位都填一遍测试，或者参考你的 StateProcessor
    # 你的 global_numeric 是 18 维
    global_numeric = torch.zeros(1, config.numeric_global_dim)
    
    # 【关键】设置能量值
    # 假设你代码里是 index 4 或者 index X。
    # 为了保险，我们不仅改 index 4，我们在一个范围内都改一下，确保肯定命中能量所在的维度
    # 或者如果你非常确定是第4位：
    ENERGY_INDEX = 4 
    global_numeric[0, ENERGY_INDEX] = energy_val 

    # 2. 构造其他空的 Tensor (全0 padding)
    # 必须符合 model.py 里的维度定义
    return SpireState(
        global_numeric = global_numeric,
        action_mask    = torch.ones(1, config.num_action_types), # 允许所有动作，方便看Q值
        
        # 简单列表
        deck_ids         = torch.zeros(1, 100, dtype=torch.long),
        draw_pile_ids    = torch.zeros(1, 100, dtype=torch.long),
        discard_pile_ids = torch.zeros(1, 100, dtype=torch.long),
        exhaust_pile_ids = torch.zeros(1, 100, dtype=torch.long),
        # limbo_ids        = torch.zeros(1, 10, dtype=torch.long),
        
        # 复杂实体
        hand_ids    = torch.zeros(1, 10, dtype=torch.long),
        hand_feats  = torch.zeros(1, 10, config.feat_dim_card),
        
        relic_ids   = torch.zeros(1, 25, dtype=torch.long),
        relic_feats = torch.zeros(1, 25, config.feat_dim_relic),
        
        potion_ids   = torch.zeros(1, 5, dtype=torch.long),
        potion_feats = torch.zeros(1, 5, config.feat_dim_potion),
        
        choice_ids   = torch.zeros(1, 15, dtype=torch.long),
        card_in_play_id = torch.zeros(1, 1, dtype=torch.long),
        
        # 玩家
        player_numeric     = torch.zeros(1, config.numeric_player_dim),
        player_power_ids   = torch.zeros(1, 20, dtype=torch.long),
        player_power_feats = torch.zeros(1, 20, config.feat_dim_power),
        player_orb_ids     = torch.zeros(1, 10, dtype=torch.long),
        player_orb_vals    = torch.zeros(1, 10, config.feat_dim_orb),
        
        # 怪物
        monster_ids         = torch.zeros(1, 5, dtype=torch.long),
        monster_intent_ids  = torch.zeros(1, 5, dtype=torch.long),
        monster_numeric     = torch.zeros(1, 5, config.numeric_monster_dim),
        monster_power_ids   = torch.zeros(1, 5, 20, dtype=torch.long),
        monster_power_feats = torch.zeros(1, 5, 20, config.feat_dim_power),
        
        # 屏幕与地图
        screen_type_val   = torch.zeros(1, 1, dtype=torch.long),
        screen_misc       = torch.zeros(1, config.dim_screen_misc),
        screen_item_ids   = torch.zeros(1, config.max_screen_items, dtype=torch.long),
        screen_item_feats = torch.zeros(1, config.max_screen_items, config.dim_screen_item_feat),
        
        map_node_ids      = torch.zeros(1, 60, dtype=torch.long),
        map_node_coords   = torch.zeros(1, 60, 2),
        map_mask          = torch.zeros(1, 60),
    )

def debug_energy_sensitivity():
    # 1. 初始化配置
    # 确保这里的维度和你训练时的一致！
    config = SpireConfig(
        numeric_global_dim=18,  # 你之前扩容过的
        numeric_monster_dim=9,
        numeric_player_dim=5
    )
    
    # 2. 加载模型
    print("正在加载模型...")
    agent = SpireAgent(config, device="cpu") # 调试用CPU即可
    model_path = os.path.join(get_root_dir(), "models", "latest.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        return
        
    # 强制加载
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # strict=False 允许忽略不匹配的层
    # 这样 global_num_enc 会保持随机初始化（复活），而其他层（Embedding等）依然加载旧权重
    keys = agent.policy_net.load_state_dict(checkpoint['model'], strict=False)

    print(f"⚠️ 未加载的层 (应该是 global/player num enc): {keys.missing_keys}")
    print(f"🗑️ 丢弃的旧层: {keys.unexpected_keys}")

    print("🔪 [Debug脚本] 正在手动重置 SharedBody 入口层...")
    def reset_layer(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
    agent.policy_net.shared_body[0].apply(reset_layer)
    print("✅ SharedBody 重置完成，链路已打通。")
    print("✅ 模型加载成功！")

    # 3. 检查权重 (直接看 Global Numeric 层)
    print("\n--- 1. 权重检查 ---")
    w = agent.policy_net.global_num_enc[0].weight
    # 假设能量是 index 4
    energy_w = w[:, 4]
    print(f"Global Numeric Layer Shape: {w.shape}")
    print(f"Index 4 (Energy) Weight Mean: {energy_w.mean().item():.6f}")
    print(f"Index 4 (Energy) Weight Std : {energy_w.std().item():.6f}")
    
    if energy_w.abs().sum() == 0:
        print("🔴 警告：能量对应的权重全是 0！这就是问题所在！")
    else:
        print("🟢 权重非零，理论上应该有反应。")

    # 4. 对比测试
    print("\n--- 2. 敏感度测试 ---")
    
    # 状态 A: 正常能量 (比如 0.6 代表 3费)
    state_a = create_dummy_state(config, energy_val=0.6)
    # 状态 B: 巨大能量 (比如 100.0)
    state_b = create_dummy_state(config, energy_val=100.0)
    
    with torch.no_grad():
        # === 探针测试 ===
        # 1. 看看 Encoder 层的输出变没变 (这是第一站)
        enc_a = agent.policy_net.global_num_enc(state_a.global_numeric)
        enc_b = agent.policy_net.global_num_enc(state_b.global_numeric)
        
        enc_diff = (enc_a - enc_b).abs().sum().item()
        
        print(f"\n[🔬 显微镜检测]")
        print(f"Encoder Output Diff: {enc_diff:.6f}")
        
        if enc_diff > 1.0:
            print("  ✅ Encoder 反应剧烈！眼睛是好的！")
        else:
            print("  ❌ Encoder 没反应？")

        out_a = agent.policy_net(state_a)
        out_b = agent.policy_net(state_b)
        
        # 取 Action Type: PLAY (假设 index 0) 的 Q 值
        q_a = out_a.q_action_type[0, 0].item()
        q_b = out_b.q_action_type[0, 0].item()
        
    print(f"Q_Action(Play) [Energy=0.6]: {q_a:.10f}")
    print(f"Q_Action(Play) [Energy=100]: {q_b:.10f}")
    print(f"Diff: {abs(q_a - q_b):.10f}")
    
    if abs(q_a - q_b) < 1e-10:
        print("🔴 结论：模型对能量变化毫无反应！(Dead Neuron / Disconnected)")
    else:
        print("🟢 结论：模型有反应！逻辑链路是通的。")

if __name__ == "__main__":
    debug_energy_sensitivity()