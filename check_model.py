import torch
import os
from spirecomm.utils.path import get_root_dir

# --- 配置你要检查的文件名 ---
FILENAME = "step_452000.pth" 
# FILENAME = "latest.pth" # 或者是这个，看你想查哪个

def inspect_checkpoint():
    file_path = os.path.join(get_root_dir(), "models", FILENAME)

    if not os.path.exists(file_path):
        print(f"❌ 找不到文件: {file_path}")
        return

    print(f"📂 正在读取: {file_path} ...")
    
    try:
        # weights_only=False 以兼容可能存在的自定义类
        checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)

        # 先打印 training_steps / total_steps 信息
        print("\n" + "="*40)
        print("       训练步数信息 (training_steps)")
        print("="*40)
        ts = checkpoint.get("training_steps", None)
        legacy_ts = checkpoint.get("total_steps", None)
        if ts is not None:
            print(f"✅ training_steps: {ts}")
        else:
            print("⚠️ 未找到 training_steps 字段")
        if legacy_ts is not None:
            print(f"(兼容字段) total_steps: {legacy_ts}")

        if 'model' not in checkpoint:
            print("❌ 文件中没有 'model' 键，可能不是有效的 checkpoint。")
            return
            
        state_dict = checkpoint['model']
        
        print("\n" + "="*40)
        print("       关键层检查 (Shared Body)")
        print("="*40)
        
        target_key = "shared_body.0.weight"
        
        if target_key in state_dict:
            weight = state_dict[target_key]
            shape = weight.shape
            print(f"🎯 层名称: {target_key}")
            print(f"📏 维度: {shape}")
            
            # 自动判断逻辑
            input_dim = shape[1] # [Output, Input]
            
            if input_dim == 1920:
                print("\n✅ [判定]: 這是 **新模型 (1920)**。")
                print("   包含: DrawPile(128) + ExhaustPile(128)。")
                print("   可以直接运行新的 trainer.py。")
            elif input_dim == 1664:
                print("\n⚠️ [判定]: 這是 **旧模型 (1664)**。")
                print("   缺失: DrawPile 和 ExhaustPile。")
                print("   需要运行修复脚本进行扩容。")
            else:
                print(f"\n❓ [判定]: 未知维度 ({input_dim})。")
        else:
            print(f"❌ 未找到 {target_key} 层，模型结构可能不同。")

        print("\n" + "-"*40)
        print("       其他层维度预览 (前10个)")
        print("-"*40)
        count = 0
        for key, value in state_dict.items():
            print(f"{key}: {value.shape}")
            count += 1
            if count >= 10:
                print("... (其余省略)")
                break

    except Exception as e:
        print(f"❌ 读取发生错误: {e}")

if __name__ == "__main__":
    inspect_checkpoint()