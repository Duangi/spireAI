import torch
import torch.nn as nn
import os
from spirecomm.utils.path import get_root_dir

# --- 配置：指定你要手术的文件名 ---
FILENAME = "latest.pth" 

def deep_surgical_reset():
    path = os.path.join(get_root_dir(), "models", FILENAME)
    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return

    print(f"📂 正在加载模型执行深度复苏手术: {FILENAME}...")
    # weights_only=False 必须加，因为里面存有 SpireConfig 对象
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']

    # 1. 定义要彻底重置的逻辑层（这些层负责“想”和“做”）
    # 我们把数值编码器也重置了，让它们配合新的 LeakyReLU 重新学习
    logic_layers = [
        "shared_body",      # 大脑主干
        "value_head",       # 状态评估头
        "action_type_head", # 动作决策头
        "scorers",          # 分支打分器
        "global_num_enc",   # 全局数值眼睛
        "player_num_enc",   # 玩家数值眼睛
        "monster_num_proj", # 怪物数值眼睛
        "pooler"            # 聚合特征层
    ]

    # 2. 定义绝对要保留的记忆层（这些层负责“认得东西”）
    # 只要名字里带 emb 的，都是我们花几十万步练出来的词汇量
    print("🧠 正在保护核心记忆层 (Embeddings)...")

    reset_count = 0
    keep_count = 0

    # 遍历 state_dict 执行手术
    for key in list(state_dict.keys()):
        # 判定是否属于逻辑层
        should_reset = any(layer_name in key for layer_name in logic_layers)
        
        if should_reset:
            param = state_dict[key]
            
            if "weight" in key:
                # 使用 Kaiming Normal 初始化，专为 LeakyReLU 设计
                # a=0.01 是因为我们用的是 LeakyReLU(0.01)
                if len(param.shape) >= 2:
                    nn.init.kaiming_normal_(param, a=0.01, mode='fan_in', nonlinearity='leaky_relu')
                else:
                    nn.init.normal_(param, std=0.02)
            elif "bias" in key:
                # 偏置项初始化为 0
                nn.init.constant_(param, 0)
            
            # 特殊处理：LayerNorm 的参数不能乱初始化
            if "LayerNorm" in key or "layer_norm" in key:
                if "weight" in key: nn.init.constant_(param, 1.0)
                if "bias" in key: nn.init.constant_(param, 0)
                
            reset_count += 1
        else:
            keep_count += 1

    # 3. 清理残留数据
    # 必须删除旧的优化器状态，因为逻辑层权重全变了，旧的动量会误导新训练
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
        print("🧹 已清除旧优化器动量数据 (重要)。")

    # 4. 保存
    save_path = path # 或者你可以改名叫 latest_reset.pth
    # 备份一下防止万一
    shutil_path = path + ".bak"
    import shutil
    shutil.copyfile(path, shutil_path)
    
    torch.save(checkpoint, save_path)
    
    print("\n" + "="*40)
    print(f"✅ 手术圆满完成！")
    print(f"   - 重置逻辑层参数: {reset_count} 个")
    print(f"   - 保留记忆层参数: {keep_count} 个")
    print(f"   - 备份文件已存至: {shutil_path}")
    print(f"🚀 模型已回归“有常识、没逻辑”的纯净状态。")
    print("="*40)

if __name__ == "__main__":
    deep_surgical_reset()