import torch
import os
import time
import uuid

# 目录配置
DATA_DIR = "./buffer_data"
MODEL_DIR = "./models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def save_experience_batch(transitions):
    """
    Worker 调用：把一批经验存到硬盘。
    transitions: list of (state, action, reward, next, done)
    """
    if not transitions:
        return
    
    # 使用 UUID 生成唯一文件名，防止冲突
    filename = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}.pt"
    filepath = os.path.join(DATA_DIR, filename)
    
    # 使用 torch.save 保存，因为它处理 Tensor 很方便
    # 为了防止写入时被读取，先写临时文件再重命名
    temp_path = filepath + ".tmp"
    torch.save(transitions, temp_path)
    os.rename(temp_path, filepath) # 原子操作，安全

def load_latest_model(agent, device):
    """
    Worker/Trainer 调用：尝试加载最新的模型
    """
    model_path = os.path.join(MODEL_DIR, "latest.pth")
    if os.path.exists(model_path):
        try:
            # map_location 确保 Worker 可以用 CPU 加载
            checkpoint = torch.load(model_path, map_location=device)
            agent.policy_net.load_state_dict(checkpoint['model'])
            return True
        except Exception as e:
            print(f"Loading model failed (might be writing): {e}")
            return False
    return False