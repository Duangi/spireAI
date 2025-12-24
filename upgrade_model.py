import torch
import os
import shutil
from spirecomm.utils.path import get_root_dir

# 指定你要修复的文件（根据你的输出，是 latest_upgraded.pth 或者 latest.pth）
# 这里我们直接修复 latest.pth，一步到位
TARGET_FILE_NAME = "latest.pth" 

def force_expand_weights():
    models_dir = os.path.join(get_root_dir(), "models")
    target_path = os.path.join(models_dir, TARGET_FILE_NAME)
    
    if not os.path.exists(target_path):
        print(f"❌ 找不到文件: {target_path}")
        return

    print(f"🔪 正在对 {TARGET_FILE_NAME} 进行强制扩容手术...")
    
    # 1. 加载 (不依赖任何自定义类)
    checkpoint = torch.load(target_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # 2. 锁定目标层
    key = "shared_body.0.weight"
    if key not in state_dict:
        print(f"❌ 严重错误：找不到层 {key}")
        return

    old_weight = state_dict[key]
    out_dim, in_dim = old_weight.shape
    print(f"   当前维度: [{out_dim}, {in_dim}]")

    # 3. 强制扩容逻辑
    if in_dim == 1664:
        print("⚡ 确认是旧维度 (1664)，开始注入新神经元...")
        
        # 目标是 1920，差值 256
        diff = 1920 - 1664
        
        # 生成随机噪声 (模拟初始化)
        extension = torch.randn(out_dim, diff) * 0.01
        
        # 暴力拼接
        # [1024, 1664] + [1024, 256] -> [1024, 1920]
        new_weight = torch.cat([old_weight, extension], dim=1)
        
        # 替换回字典
        state_dict[key] = new_weight
        
        # 还要记得保存回 checkpoint
        checkpoint['model'] = state_dict
        
        # 4. 覆盖保存
        torch.save(checkpoint, target_path)
        print(f"✅ 手术成功！文件已覆盖: {target_path}")
        print(f"   新维度: {new_weight.shape}")
        
    elif in_dim == 1920:
        print("✅ 该文件已经是 1920 维了，不需要手术。")
    else:
        print(f"❓ 奇怪的维度 {in_dim}，脚本不敢乱动。")

if __name__ == "__main__":
    force_expand_weights()