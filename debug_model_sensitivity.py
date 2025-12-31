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
    制造一维 Tensor，并自动适配 Config 里的维度定义。
    """
    # 自动从 config 读取维度，防止报错
    global_numeric = torch.zeros(config.numeric_global_dim)
    player_numeric = torch.zeros(config.numeric_player_dim)
    
    # 设置能量
    ENERGY_INDEX = 4 
    player_numeric[ENERGY_INDEX] = energy_val 

    return SpireState(
        global_numeric = global_numeric,
        player_numeric = player_numeric,
        action_mask    = torch.ones(config.num_action_types, dtype=torch.bool),
        deck_ids         = torch.zeros(100, dtype=torch.long),
        draw_pile_ids    = torch.zeros(100, dtype=torch.long),
        discard_pile_ids = torch.zeros(100, dtype=torch.long),
        exhaust_pile_ids = torch.zeros(100, dtype=torch.long),
        hand_ids    = torch.zeros(10, dtype=torch.long),
        hand_feats  = torch.zeros(10, 16),
        relic_ids   = torch.zeros(25, dtype=torch.long),
        relic_feats = torch.zeros(25, 3),
        potion_ids   = torch.zeros(5, dtype=torch.long),
        potion_feats = torch.zeros(5, 2),
        choice_ids   = torch.zeros(15, dtype=torch.long),
        card_in_play_id = torch.zeros(1, dtype=torch.long),
        player_power_ids   = torch.zeros(20, dtype=torch.long),
        player_power_feats = torch.zeros(20, 3),
        player_orb_ids     = torch.zeros(10, dtype=torch.long),
        player_orb_vals    = torch.zeros(10, 2),
        monster_ids         = torch.zeros(5, dtype=torch.long),
        monster_intent_ids  = torch.zeros(5, dtype=torch.long),
        monster_numeric     = torch.zeros(5, 9),
        monster_power_ids   = torch.zeros(5, 20, dtype=torch.long),
        monster_power_feats = torch.zeros(5, 20, 3),
        screen_type_val   = torch.zeros(1, dtype=torch.long),
        screen_misc       = torch.zeros(8),
        # 【自动适配】使用 config 里的定义
        screen_item_ids   = torch.zeros(config.max_screen_items, dtype=torch.long),
        screen_item_feats = torch.zeros(config.max_screen_items, config.dim_screen_item_feat),
        map_node_ids      = torch.zeros(60, dtype=torch.long),
        map_node_coords   = torch.zeros(60, 2),
        map_mask          = torch.zeros(60)
    )

def debug_energy_sensitivity():
    # 1. 这里的维度必须和你 model.py 里的当前定义一致！
    config = SpireConfig(
        numeric_global_dim=18,
        numeric_monster_dim=9,
        numeric_player_dim=5,
        dim_screen_item_feat=17 # 假设你代码里改成了17
    )
    
    print("正在加载模型...")
    agent = SpireAgent(config, device="cpu")
    model_path = os.path.join(get_root_dir(), "models", "latest.pth")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    # 使用 strict=False。如果文件里是 16 维，代码是 17 维，它会自动跳过不加载
    agent.policy_net.load_state_dict(checkpoint['model'], strict=False)

    print("🔪 [Debug脚本] 正在手动重置 SharedBody 入口层...")
    def reset_layer(layer):
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    agent.policy_net.shared_body[0].apply(reset_layer)
    print("✅ 模型加载并手术成功！")

    # 3. 对比测试
    print("\n--- 2. 敏感度测试 ---")
    state_a = create_dummy_state(config, energy_val=0.6)
    state_b = create_dummy_state(config, energy_val=10.0) # 调大点
    
    batch_a = agent.collate_states([state_a])
    batch_b = agent.collate_states([state_b])

    with torch.no_grad():
        # 【探针 1】
        enc_a = agent.policy_net.player_num_enc(batch_a.player_numeric)
        enc_b = agent.policy_net.player_num_enc(batch_b.player_numeric)
        enc_diff = (enc_a - enc_b).abs().sum().item()
        print(f"[🔬 显微镜检测] Player Encoder Output Diff: {enc_diff:.6f}")

        # 【预测 Q 值】
        # 这里之前会报错，因为 screen_item_feats 维度没加载对
        # 只要用了这个修复脚本，应该就能跑到底
        out_a = agent.policy_net(batch_a)
        out_b = agent.policy_net(batch_b)
        
        q_a = out_a.q_action_type[0, 0].item()
        q_b = out_b.q_action_type[0, 0].item()
        
    print(f"Q_Action(Play) [Energy=0.6]: {q_a:.10f}")
    print(f"Q_Action(Play) [Energy=10.0]: {q_b:.10f}")
    print(f"Diff: {abs(q_a - q_b):.10f}")

if __name__ == "__main__":
    debug_energy_sensitivity()