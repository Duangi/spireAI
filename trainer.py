import os
import time
import glob
import shutil
import torch
import sys
import re
from datetime import datetime

import wandb
from spirecomm.ai.constants import MAX_DECK_SIZE
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.utils.path import get_root_dir

# --- Configuration ---
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
MEMORY_REMOTE_DIR = os.path.join(get_root_dir(), "data", "memory_remote")
ARCHIVE_DIR = os.path.join(get_root_dir(), "data", "archive")
MODELS_DIR = os.path.join(get_root_dir(), "models")

os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(MEMORY_REMOTE_DIR, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
BATCH_SIZE = 256

SAVE_INTERVAL = 1000  # 每训练多少步保存一次模型
TARGET_UPDATE_INTERVAL = 1000  # 每训练多少步更新一次目标网络
RR = 8  # 根据经验池数据量动态调整训练次数的比例因子

def get_latest_model_path(player_class=None):
    target_dir = MODELS_DIR
    if player_class:
        class_dir = os.path.join(MODELS_DIR, player_class.name)
        if os.path.exists(class_dir):
            target_dir = class_dir
            
    if not os.path.exists(target_dir):
        return None, 0

    # Logic to find the model with the highest step number
    # Format: step_{step}.pth
    model_files = [f for f in os.listdir(target_dir) if f.startswith("step_") and f.endswith(".pth")]
    
    latest_step = 0
    latest_model_path = None
    if len(model_files) == 0:
        return None, 0

    for f in model_files:
        try:
            step_num = int(f[len("step_"):-len(".pth")])
            if step_num > latest_step:
                latest_step = step_num
                latest_model_path = os.path.join(target_dir, f)
        except ValueError:
            continue
            
    if latest_model_path:
        return latest_model_path, latest_step
    else:
        return None, 0

# ========================================================
# 【新增】兼容补丁函数
# ========================================================
def fix_legacy_state(state):
    """
    终极数据修复补丁：
    不仅补全缺失字段，还强制将旧的维度(10)对齐到新维度(15)。
    """
    MAX_DECK_LEN = 100
    TARGET_MONSTER_COUNT = 15  # 你现在设定的新上限
    MONSTER_NUMERIC_DIM = 9   # 怪物数值特征维度
    POWER_COUNT = 20          # 怪物Buff上限

    # 1. 修复 Draw/Exhaust 缺失 (保持原样)
    if not hasattr(state, 'draw_pile_ids'):
        state.draw_pile_ids = torch.zeros(MAX_DECK_LEN, dtype=torch.long)
    if not hasattr(state, 'exhaust_pile_ids'):
        state.exhaust_pile_ids = torch.zeros(MAX_DECK_LEN, dtype=torch.long)

    # 2. 【核心修复】修复怪物维度不匹配 (10 -> 15)
    # 检查 monster_ids 的长度
    current_m_count = state.monster_ids.shape[0]
    
    if current_m_count < TARGET_MONSTER_COUNT:
        diff = TARGET_MONSTER_COUNT - current_m_count
        
        # A. 补齐 ID 类 (LongTensor)
        state.monster_ids = torch.cat([state.monster_ids, torch.zeros(diff, dtype=torch.long)])
        state.monster_intent_ids = torch.cat([state.monster_intent_ids, torch.zeros(diff, dtype=torch.long)])
        
        # B. 补齐数值类 (FloatTensor [N, 9])
        m_num_padding = torch.zeros((diff, MONSTER_NUMERIC_DIM), dtype=torch.float32)
        state.monster_numeric = torch.cat([state.monster_numeric, m_num_padding], dim=0)
        
        # C. 补齐 Power IDs (LongTensor [N, 20])
        m_pow_id_padding = torch.zeros((diff, POWER_COUNT), dtype=torch.long)
        state.monster_power_ids = torch.cat([state.monster_power_ids, m_pow_id_padding], dim=0)
        
        # D. 补齐 Power Feats (FloatTensor [N, 20, 3])
        m_pow_feat_padding = torch.zeros((diff, POWER_COUNT, 3), dtype=torch.float32)
        state.monster_power_feats = torch.cat([state.monster_power_feats, m_pow_feat_padding], dim=0)

    # 3. 修复 Global Numeric (17 -> 18) 如果之前没改彻底的话
    if state.global_numeric.shape[-1] < 18:
        diff_g = 18 - state.global_numeric.shape[-1]
        g_padding = torch.zeros((*state.global_numeric.shape[:-1], diff_g), dtype=torch.float32)
        state.global_numeric = torch.cat([state.global_numeric, g_padding], dim=-1)

    return state

def manage_archive(archive_dir, max_files=10000, max_gb=20):
    """
    管理归档文件夹：如果文件数量或总大小超过限制，则删除最早的文件。
    :param archive_dir: 归档文件夹路径
    :param max_files: 最大保留文件数
    :param max_gb: 最大占用磁盘空间 (GB)
    """
    # 1. 获取所有 .pt 文件及其元数据
    file_list = []
    for root, _, filenames in os.walk(archive_dir):
        for f in filenames:
            if f.endswith(".pt"):
                path = os.path.join(root, f)
                try:
                    stats = os.stat(path)
                    # 记录 (文件路径, 修改时间, 文件大小)
                    file_list.append({
                        'path': path,
                        'time': stats.st_mtime,
                        'size': stats.st_size
                    })
                except OSError:
                    continue

    if not file_list:
        return

    # 2. 按时间排序 (从旧到新)
    # 确保我们删掉的是最早产出的数据
    file_list.sort(key=lambda x: x['time'])

    total_count = len(file_list)
    total_size_bytes = sum(f['size'] for f in file_list)
    max_size_bytes = max_gb * 1024 * 1024 * 1024

    # 3. 循环删除直到满足条件
    files_deleted = 0
    while (total_count > max_files or total_size_bytes > max_size_bytes) and file_list:
        target = file_list.pop(0) # 弹出最老的文件
        try:
            os.remove(target['path'])
            total_count -= 1
            total_size_bytes -= target['size']
            files_deleted += 1
        except OSError as e:
            print(f"清理失败: {target['path']}, 错误: {e}")

    if files_deleted > 0:
        print(f"清理归档: 删除了 {files_deleted} 个旧文件。当前占用: {total_size_bytes/1024**3:.2f} GB")

def run_trainer():
    # Initialize WandB
    if wandb.run is None:
        wandb_logger = WandbLogger(project_name="spire-ai-trainer", run_name=f"Trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Initialize Agent
    agent = DQNAgent(play_mode=False, wandb_logger=wandb_logger)
    
    # Load latest model
    latest_model_path, initial_step = get_latest_model_path()
    current_step = 0
    
    if latest_model_path:
        print(f"Loading latest model from {latest_model_path} (Step: {initial_step})...")
        agent.load_model(latest_model_path)
        # training_steps 保持 checkpoint 中的值，温度从对应进度继续
        current_step = agent.dqn_algorithm.training_steps
    else:
        print("No existing model found. Starting from scratch.")

    print(f"Trainer started. Current Step: {current_step}")
    print(f"Monitoring {MEMORY_DIR} for new data...")
    print(f"Monitoring {MEMORY_REMOTE_DIR} for new data...")
    agent.dqn_algorithm.reload_config_from_file()
    # Main Loop
    while True:
        # 1. Scan for memory files
        # 递归查找 data/memory 下的所有 .pt 文件（同时也检查 data/memory_remote）
        pattern_local = os.path.join(MEMORY_DIR, "**", "*.pt")
        pattern_remote = os.path.join(MEMORY_REMOTE_DIR, "**", "*.pt")
        files_steps = 0 # 读取文件的第二个数字，表示该文件对应的 step
        # 按文件名中的 step_xxx 数字大小排序，确保先训练较早的数据
        local_files = glob.glob(pattern_local, recursive=True)
        remote_files = glob.glob(pattern_remote, recursive=True)
        all_files = local_files + remote_files
        def sort_key_by_step(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'step_(\d+)', filename)
            if match:
                return int(match.group(1))
            return float('inf')
        memory_files = sorted(all_files, key=sort_key_by_step)
        batch_process_files = memory_files[:50]

        if not batch_process_files:
            # 如果没有文件，稍微等待一下，避免空转占用CPU
            # print("No data files found. Waiting...", end='\r')
            time.sleep(1)
        else:
            print(f"\nFound {len(batch_process_files)} data files. Processing...")

            files_processed = 0

            for filepath in batch_process_files:
                try:
                    # Load data
                    # weights_only=False is required for custom objects like SpireState
                    # 我们信任 worker 产生的数据，所以这里设置为 False
                    transitions = torch.load(filepath, weights_only=False)
                    # 文件的命名格式：step_{step}_{game_steps}_{timestamp}.pt
                    parts = os.path.basename(filepath).split('_')
                    if len(parts) >= 3:
                        try:
                            files_steps += int(parts[2])
                        except ValueError:
                            pass
                    # Add to replay buffer
                    for t in transitions:
                        # ========================================================
                        # 【新增】调用兼容函数，原地修复旧数据
                        # ========================================================
                        fix_legacy_state(t['state_tensor'])
                        fix_legacy_state(t['next_state_tensor'])
                        # ========================================================

                        agent.dqn_algorithm.remember(
                            t['state_tensor'],
                            t['action'],
                            t['reward'],
                            t['next_state_tensor'],
                            t['done'],
                            t['reward_details']
                        )

                    # --- Episode-level WandB 指标 ---
                    ep_return = sum(t['reward'] for t in transitions)
                    ep_length = len(transitions)
                    ep_floor = 0
                    # 从最后一步的 next_state 提取楼层 (global_numeric[3] * 60)
                    try:
                        last_next = transitions[-1]['next_state_tensor']
                        if hasattr(last_next, 'global_numeric'):
                            ep_floor = int(last_next.global_numeric[3].item() * 60)
                    except Exception:
                        pass
                    if wandb.run is not None:
                        wandb.log({
                            "episode/return":        ep_return,
                            "episode/length":        ep_length,
                            "episode/floor_reached": ep_floor,
                        }, step=current_step)
                    
                    # Move to archive
                    # 保持目录结构移动到 archive
                    # e.g. data/memory/DEFECT/file.pt -> data/archive/DEFECT/file.pt
                    try:
                        abs_filepath = os.path.abspath(filepath)
                        abs_memory = os.path.abspath(MEMORY_DIR)
                        abs_remote = os.path.abspath(MEMORY_REMOTE_DIR)
                        if os.path.commonpath([abs_filepath, abs_memory]) == abs_memory:
                            rel_path = os.path.relpath(filepath, MEMORY_DIR)
                        elif os.path.commonpath([abs_filepath, abs_remote]) == abs_remote:
                            rel_path = os.path.relpath(filepath, MEMORY_REMOTE_DIR)
                        else:
                            # Fallback: place under archive root with basename
                            rel_path = os.path.basename(filepath)
                    except Exception:
                        rel_path = os.path.basename(filepath)

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
            print(f"Total steps from files: {files_steps}")

        # 2. Train
        # 只要经验池里的数据够一个 Batch，就开始训练
        # 每次循环训练一定次数，或者根据数据量动态调整
        min_steps_to_train = BATCH_SIZE // RR
        if files_steps > min_steps_to_train and len(agent.dqn_algorithm.memory) >= BATCH_SIZE:
            # 根据files_steps调整训练次数，要求RR达到8
            # 训练次数 = 新数据量 * RR / BATCH_SIZE
            target_train_loops = max(files_steps * RR // BATCH_SIZE, 1)
            print(f"此次训练 {target_train_loops} 次 (基于 {files_steps} 步新数据)")
            for _ in range(target_train_loops):
                agent.dqn_algorithm.train()
                current_step += 1
                
                agent.dqn_algorithm.update_target_net(soft=True, tau=0.005)
                
                # Save Model
                if current_step % SAVE_INTERVAL == 0:
                    save_path = os.path.join(MODELS_DIR, f"step_{current_step}.pth")
                    agent.save_model(save_path)
                    print(f"Saved model to {save_path}")
                    
                    # 更新 latest 副本，方便 worker 快速找到（虽然 worker 也会扫文件夹）
                    latest_path = os.path.join(MODELS_DIR, "latest.pth")
                    try:
                        shutil.copyfile(save_path, latest_path)
                    except Exception:
                        pass

                    # 定期清理
                    manage_archive(ARCHIVE_DIR, max_files=20000, max_gb=100)
                if current_step % TARGET_UPDATE_INTERVAL == 0:
                    agent.dqn_algorithm.reload_config_from_file()

if __name__ == "__main__":
    run_trainer()