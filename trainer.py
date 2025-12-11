import os
import time
import pickle
import glob
import torch
import wandb
from datetime import datetime
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.utils.path import get_root_dir

# --- Configuration ---
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
MODELS_DIR = os.path.join(get_root_dir(), "models")
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

BATCH_SIZE = 32

def run_trainer(target_model_step=0, data_range_min=0, data_range_max=1000000000):
    # Initialize WandB
    wandb_logger = WandbLogger(project_name="spire-ai-trainer", run_name=f"Trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Initialize Agent (Training Mode)
    agent = DQNAgent(play_mode=False, wandb_logger=wandb_logger)
    
    # Load initial model
    if target_model_step > 0:
        model_path = os.path.join(MODELS_DIR, f"dqn_model_step_{target_model_step}.pth")
        if os.path.exists(model_path):
            print(f"Loading initial model from {model_path}...")
            agent.load_model(model_path)
        else:
            print(f"Warning: Model step {target_model_step} not found. Starting from scratch.")
    
    print(f"Trainer started. Target Model Step: {target_model_step}. Data Range: {data_range_min}-{data_range_max}")
    
    current_step = target_model_step
    
    while True:
        # 1. Scan for memory files recursively
        all_files = sorted(glob.glob(os.path.join(MEMORY_DIR, "**", "*.pt"), recursive=True))
        
        # Filter files based on data range
        valid_files = []
        for f in all_files:
            basename = os.path.basename(f)
            # Expected format: step_{base_step}_{game_steps}_{timestamp}.pt
            try:
                parts = basename.split('_')
                if len(parts) >= 4 and parts[0] == 'step':
                    base_step = int(parts[1])
                    if data_range_min <= base_step <= data_range_max:
                        valid_files.append(f)
            except ValueError:
                continue
        
        if not valid_files:
            print("No valid data files found in range. Waiting...")
            time.sleep(5)
            continue
            
        # 2. Process files
        for filepath in valid_files:
            try:
                transitions = torch.load(filepath)
                
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
                
                print(f"Loaded {len(transitions)} transitions from {os.path.basename(filepath)}")
                
                # Remove processed file
                os.remove(filepath)
                
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                try:
                    os.remove(filepath)
                except:
                    pass

        # 3. Train
        # Calculate how many full batches we can train
        num_samples = len(agent.dqn_algorithm.memory)
        if num_samples >= BATCH_SIZE:
            num_batches = num_samples // BATCH_SIZE
            
            print(f"Training on {num_batches} batches...")
            
            for _ in range(num_batches):
                agent.dqn_algorithm.train()
                current_step += 1
            
            # Save new model
            save_model(agent, current_step)
            
            # Clear memory after training to avoid reusing data? 
            # Or keep it for experience replay? 
            # User said "discard extra part", implying we consume data.
            # But standard DQN keeps a buffer. 
            # Given the "offline" nature described, let's clear it to strictly follow "train on this data range".
            # Actually, user said "yyy-xxx should be divisible by batch_size", which implies strict consumption.
            agent.dqn_algorithm.memory.clear()

def save_model(agent, step):
    filename = f"dqn_model_step_{step}.pth"
    save_path = os.path.join(MODELS_DIR, filename)
    
    # Save to temp file then rename for atomic update
    temp_path = save_path + ".tmp"
    try:
        torch.save(agent.dqn_algorithm.policy_net.state_dict(), temp_path)
        if os.path.exists(save_path):
            os.remove(save_path)
        os.rename(temp_path, save_path)
        print(f"Model saved to {save_path}")
        
        # Also update latest for convenience if needed, but worker looks for step_xxx
        latest_path = os.path.join(MODELS_DIR, "dqn_model_latest.pth")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        # torch.save(agent.dqn_algorithm.policy_net.state_dict(), latest_path) 
        # Actually, let's just copy or symlink if we wanted, but worker logic is updated to look for step_xxx
        
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Example usage:
    # python trainer.py --step 1000 --min 1000 --max 2000
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0, help="Initial model step to load")
    parser.add_argument("--min", type=int, default=0, help="Min data step range")
    parser.add_argument("--max", type=int, default=1000000000, help="Max data step range")
    args = parser.parse_args()
    
    run_trainer(args.step, args.min, args.max)