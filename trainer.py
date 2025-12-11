import os
import time
import glob
import shutil
import torch
import sys
from datetime import datetime
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.utils.path import get_root_dir

# --- Configuration ---
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
ARCHIVE_DIR = os.path.join(get_root_dir(), "data", "archive")
MODELS_DIR = os.path.join(get_root_dir(), "models")

os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
BATCH_SIZE = 32
MIN_MEMORY_TO_TRAIN = BATCH_SIZE*4

SAVE_INTERVAL = BATCH_SIZE*32  # 每训练多少步保存一次模型
TARGET_UPDATE_INTERVAL = BATCH_SIZE*4 # 每训练多少步更新一次目标网络
TRAIN_BATCHES_PER_LOOP = BATCH_SIZE*4 # 每次循环训练多少个Batch

def get_latest_model_path():
    """Find the model with the highest step number in MODELS_DIR."""
    # 查找所有符合 dqn_model_step_{step}.pth 格式的文件
    model_files = [f for f in os.listdir(MODELS_DIR) if f.startswith("dqn_model_step_") and f.endswith(".pth")]
    
    latest_step = -1
    latest_model_path = None
    
    for f in model_files:
        try:
            # 解析文件名中的 step
            # Format: dqn_model_step_{step}.pth
            # split('_') -> ['dqn', 'model', 'step', '100.pth']
            step_part = f.split('_')[3]
            step_num = int(step_part.split('.')[0])
            
            if step_num > latest_step:
                latest_step = step_num
                latest_model_path = os.path.join(MODELS_DIR, f)
        except (ValueError, IndexError):
            continue
            
    return latest_model_path, latest_step

def run_trainer():
    # Initialize WandB
    wandb_logger = WandbLogger(project_name="spire-ai-trainer", run_name=f"Trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Initialize Agent
    agent = DQNAgent(play_mode=False, wandb_logger=wandb_logger)
    
    # Load latest model
    latest_model_path, initial_step = get_latest_model_path()
    current_step = 0
    
    if latest_model_path:
        print(f"Loading latest model from {latest_model_path} (Step: {initial_step})...")
        agent.load_model(latest_model_path)
        current_step = initial_step
    else:
        print("No existing model found. Starting from scratch.")

    print(f"Trainer started. Current Step: {current_step}")
    print(f"Monitoring {MEMORY_DIR} for new data...")
    
    # Main Loop
    while True:
        # 1. Scan for memory files
        # 递归查找 data/memory 下的所有 .pt 文件
        pattern = os.path.join(MEMORY_DIR, "**", "*.pt")
        memory_files = sorted(glob.glob(pattern, recursive=True))
        
        if not memory_files:
            # 如果没有文件，稍微等待一下，避免空转占用CPU
            # print("No data files found. Waiting...", end='\r')
            time.sleep(1)
        else:
            print(f"\nFound {len(memory_files)} data files. Processing...")
            
            files_processed = 0
            
            for filepath in memory_files:
                try:
                    # Load data
                    # weights_only=False is required for custom objects like SpireState
                    # 我们信任 worker 产生的数据，所以这里设置为 False
                    transitions = torch.load(filepath, weights_only=False)
                    
                    # Add to replay buffer
                    for t in transitions:
                        agent.dqn_algorithm.remember(
                            t['state_tensor'], 
                            t['action'], 
                            t['reward'], 
                            t['next_state_tensor'], 
                            t['done'], 
                            t['reward_details']
                        )
                    
                    # Move to archive
                    # 保持目录结构移动到 archive
                    # e.g. data/memory/DEFECT/file.pt -> data/archive/DEFECT/file.pt
                    rel_path = os.path.relpath(filepath, MEMORY_DIR)
                    archive_path = os.path.join(ARCHIVE_DIR, rel_path)
                    os.makedirs(os.path.dirname(archive_path), exist_ok=True)
                    
                    # 如果目标文件已存在（极小概率），添加时间戳避免覆盖
                    if os.path.exists(archive_path):
                        base, ext = os.path.splitext(archive_path)
                        archive_path = f"{base}_{int(time.time())}{ext}"
                    
                    shutil.move(filepath, archive_path)
                    files_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
                    # 出错的文件重命名为 .err，避免反复读取报错
                    try:
                        error_path = filepath + ".err"
                        os.rename(filepath, error_path)
                    except:
                        pass

            print(f"Processed {files_processed} files. Replay buffer size: {len(agent.dqn_algorithm.memory)}")

        # 2. Train
        # 只要经验池里的数据够一个 Batch，就开始训练
        # 每次循环训练一定次数，或者根据数据量动态调整
        if len(agent.dqn_algorithm.memory) >= BATCH_SIZE:
            # print(f"Training {TRAIN_BATCHES_PER_LOOP} batches...")
            for _ in range(TRAIN_BATCHES_PER_LOOP):
                agent.dqn_algorithm.train()
                current_step += 1
                
                # Update Target Net
                if current_step % TARGET_UPDATE_INTERVAL == 0:
                    agent.dqn_algorithm.update_target_net()
                
                # Save Model
                if current_step % SAVE_INTERVAL == 0:
                    save_path = os.path.join(MODELS_DIR, f"dqn_model_step_{current_step}.pth")
                    agent.save_model(save_path)
                    print(f"Saved model to {save_path}")
                    
                    # 更新 latest 副本，方便 worker 快速找到（虽然 worker 也会扫文件夹）
                    latest_path = os.path.join(MODELS_DIR, "dqn_model_latest.pth")
                    try:
                        shutil.copyfile(save_path, latest_path)
                    except Exception:
                        pass

if __name__ == "__main__":
    run_trainer()
