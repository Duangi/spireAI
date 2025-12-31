# check_weights.py

import torch
from spirecomm.utils.path import get_root_dir
import os

path = os.path.join(get_root_dir(), "models", "step_930000.pth") # 替换为你正在跑的模型
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
state_dict = checkpoint['model']

# 检查全局数值编码层的权重
w = state_dict['global_num_enc.weight'] # Shape [128, 18]
b = state_dict['global_num_enc.bias']

print(f"Weight Mean: {w.mean().item()}")
print(f"Weight Max:  {w.max().item()}")
print(f"Weight Min:  {w.min().item()}")
print("-" * 20)
# 检查 specifically 对应 Energy 的那一列权重 (假设 Energy 是第 4 列)
energy_col = w[:, 4] 
print(f"Energy Weights Sum: {energy_col.sum().item()}")
print(f"Energy Weights Abs Mean: {energy_col.abs().mean().item()}")

if energy_col.abs().mean().item() < 1e-6:
    print("🔴 警告：能量对应的权重全是 0！模型瞎了！")
else:
    print("🟢 权重看起来正常。")