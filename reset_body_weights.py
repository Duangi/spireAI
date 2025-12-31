import torch
import os
import shutil
from spirecomm.ai.dqn_core.model import SpireConfig, SpireDQN
from spirecomm.utils.path import get_root_dir

def reset_shared_body_offline():
    # 1. 路径配置
    models_dir = os.path.join(get_root_dir(), "models")
    latest_path = os.path.join(models_dir, "step_942000.pth")
    
    if not os.path.exists(latest_path):
        print("❌ 找不到 step_942000.pth")
        return

    print(f"📂 加载模型: {latest_path} ...")
    # weights_only=False 必须加
    checkpoint = torch.load(latest_path, map_location='cpu', weights_only=False)
    
    # 2. 实例化一个全新的模型对象 (随机初始化的)
    # 确保你的 model.py 里的 Config 默认值和文件里的一致 (18维, 9维等)
    config = SpireConfig(
        numeric_global_dim=18, 
        numeric_monster_dim=9, 
        numeric_player_dim=5
    )
    fresh_model = SpireDQN(config)
    
    # 3. 将旧权重加载进新模型 (strict=False)
    # 这时，除了形状不匹配的层（如果有），其他层都变成了旧权重
    # 但我们需要保留 shared_body.0 的随机性，所以我们需要反向操作
    
    # 更简单的逻辑：直接操作 checkpoint 字典
    state_dict = checkpoint['model']
    
    target_key_weight = "shared_body.0.weight"
    target_key_bias = "shared_body.0.bias"
    
    print(f"🔄 正在重置层: {target_key_weight}")
    
    # 从 fresh_model (全随机) 中提取这一层的权重
    fresh_weight = fresh_model.shared_body[0].weight.data
    fresh_bias = fresh_model.shared_body[0].bias.data
    
    # 覆盖 checkpoint 里的旧权重
    # 注意检查维度是否一致 (应该是 1024x1920)
    if state_dict[target_key_weight].shape == fresh_weight.shape:
        state_dict[target_key_weight] = fresh_weight
        state_dict[target_key_bias] = fresh_bias
        print("✅ 权重覆盖成功！(使用了全新的随机初始化值)")
    else:
        print(f"❌ 维度不匹配！文件:{state_dict[target_key_weight].shape} vs 代码:{fresh_weight.shape}")
        return

    # 4. 保存回去
    checkpoint['model'] = state_dict
    
    # 备份一下
    shutil.copyfile(latest_path, latest_path + ".bak_before_reset")
    
    torch.save(checkpoint, latest_path)
    print(f"💾 已保存至: {latest_path}")
    print("🚀 现在可以直接运行 trainer.py 了 (无需在代码里写 reset)")

if __name__ == "__main__":
    reset_shared_body_offline()