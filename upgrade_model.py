import torch
import os
import shutil
from spirecomm.utils.path import get_root_dir

# 指定要修复的文件 (通常是 latest.pth)
TARGET_FILENAME = "step_670000.pth"

def fix_global_numeric_dim():
    models_dir = os.path.join(get_root_dir(), "models")
    path = os.path.join(models_dir, TARGET_FILENAME)
    
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return

    print(f"🔧 正在检查: {path} ...")
    
    # 加载
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # 目标层：全局数值编码层
    key = "global_num_enc.weight"
    
    if key not in state_dict:
        print(f"❌ 找不到层: {key}")
        return

    old_weight = state_dict[key] # 应该是 [128, 17]
    out_dim, in_dim = old_weight.shape
    
    print(f"   当前维度: {old_weight.shape}")

    if in_dim == 18:
        print("✅ 此文件已经是 18 维了，无需修复！")
        return
    elif in_dim == 17:
        print("⚡ 检测到旧维度 (17)，开始扩容到 18...")
        
        # 计算差值 (1)
        diff = 18 - 17
        
        # 生成随机噪声权重 (1列)
        extension = torch.randn(out_dim, diff) * 0.01
        
        # 拼接：[旧权重, 新权重] -> [128, 18]
        new_weight = torch.cat([old_weight, extension], dim=1)
        
        # 替换回去
        state_dict[key] = new_weight
        checkpoint['model'] = state_dict
        
        # 备份并覆盖
        shutil.copyfile(path, path + ".bak_17")
        torch.save(checkpoint, path)
        
        print(f"✅ 修复完成！新维度: {new_weight.shape}")
        print("🚀 现在可以重新启动 Evaluator/Trainer 了！")
        
    else:
        print(f"❌ 未知维度 {in_dim}，未做处理。")

if __name__ == "__main__":
    fix_global_numeric_dim()