import torch
import os
from spirecomm.utils.path import get_root_dir

# --- 配置 ---
FILENAME = "step_670000.pth" 

def check_model_dimensions():
    path = os.path.join(get_root_dir(), "models", FILENAME)

    if not os.path.exists(path):
        print(f"❌ 找不到文件: {path}")
        return

    print(f"📂 正在读取模型: {path} ...")
    
    try:
        # 加载模型
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint['model']

        print("-" * 50)

        # --- 检查点 1: 全局数值层 (Global Numeric) ---
        # 目标: [128, 18]
        key_global = "global_num_enc.weight"
        if key_global in state_dict:
            w = state_dict[key_global]
            print(f"🎯 检查层: {key_global}")
            print(f"   实际维度: {w.shape}")
            
            if w.shape[1] == 18:
                print("   ✅ [通过] 已成功扩容到 18 维 (包含格挡溢出特征)。")
            elif w.shape[1] == 17:
                print("   ❌ [失败] 仍然是旧的 17 维。修复脚本可能未生效。")
            else:
                print(f"   ❓ [未知] 奇怪的维度: {w.shape[1]}")
        else:
            print(f"❌ 找不到层: {key_global}")

        print("-" * 50)

        # --- 检查点 2: 主干层 (Shared Body) ---
        # 目标: [1024, 1920] (确保之前的修复没被覆盖)
        key_body = "shared_body.0.weight"
        if key_body in state_dict:
            w = state_dict[key_body]
            print(f"🎯 检查层: {key_body}")
            print(f"   实际维度: {w.shape}")
            
            if w.shape[1] == 1920:
                print("   ✅ [通过] 维持在 1920 维 (包含抽牌/消耗堆)。")
            else:
                print(f"   ⚠️ [警告] 维度不对！期望 1920，实际 {w.shape[1]}")
        
        print("-" * 50)
        
        # --- 检查点 3: 怪物数值层 (Monster Numeric) ---
        # 目标: [128, 9] (确认你没有改动过这个)
        key_monster = "monster_num_proj.weight"
        if key_monster in state_dict:
            w = state_dict[key_monster]
            print(f"🎯 检查层: {key_monster}")
            print(f"   实际维度: {w.shape}")
            if w.shape[1] == 9:
                print("   ✅ [通过] 维度为 9。")
            else:
                print(f"   ℹ️ [提示] 维度为 {w.shape[1]} (如果你改过Monster特征这是正常的)。")

    except Exception as e:
        print(f"❌ 读取发生严重错误: {e}")

if __name__ == "__main__":
    check_model_dimensions()