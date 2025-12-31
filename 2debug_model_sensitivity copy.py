import torch
import os
from spirecomm.utils.path import get_root_dir

# 指定你要修正的文件
FILENAME = "step_952000.pth"

def sync_internal_steps():
    path = os.path.join(get_root_dir(), "models", FILENAME)
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return

    print(f"📂 正在修正文件内部步数: {FILENAME}...")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # 提取文件名里的数字
    import re
    m = re.search(r'step[_-]?(\d+)', FILENAME)
    real_steps = int(m.group(1)) if m else 0
    
    if real_steps == 0:
        print("❌ 无法从文件名解析步数。")
        return

    # 强制修改字典里的所有计数器字段
    checkpoint['training_steps'] = real_steps
    checkpoint['total_steps'] = real_steps
    
    # 顺便把优化器删了（确保维度彻底干净，反正你现在也得重新练优化器）
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
        print("🧹 已顺便清理优化器状态。")

    torch.save(checkpoint, path)
    print(f"✅ 修正成功！内部步数已设为: {real_steps}")
    print("🚀 现在重启 Trainer，温度应该会回到 1.6 左右。")

if __name__ == "__main__":
    sync_internal_steps()