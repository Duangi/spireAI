import torch
import os
from spirecomm.utils.path import get_root_dir

def reset_decision_layers():
    path = os.path.join(get_root_dir(), "models", "step_946000.pth")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # 我们要重置的所有“逻辑处理”层
    # 只要不包含 'emb' 和 'fusion' 的，基本都是我们要重置的
    keys_to_reset = [
        "shared_body",
        "value_head",
        "action_type_head",
        "scorers"
    ]

    print(f"🔪 正在切除坏死的决策神经元...")
    
    # 遍历所有权重
    for key in list(state_dict.keys()):
        if any(target in key for target in keys_to_reset):
            # 找到对应的参数
            param = state_dict[key]
            # 执行随机初始化 (Kaiming分布是目前效果最好的)
            if 'weight' in key:
                if len(param.shape) >= 2:
                    torch.nn.init.kaiming_normal_(param)
                else:
                    torch.nn.init.normal_(param)
            elif 'bias' in key:
                torch.nn.init.constant_(param, 0)
            
            print(f"  ✨ 已重置: {key}")

    # 保存并覆盖
    torch.save(checkpoint, path)
    print("\n✅ 手术完成！决策层已回归初始状态，Embedding 记忆已保留。")

if __name__ == "__main__":
    reset_decision_layers()