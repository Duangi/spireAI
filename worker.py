import os
import sys
import time
import pickle
import torch
import itertools
from datetime import datetime
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.utils.path import get_root_dir

# --- Configuration ---
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
MODELS_DIR = os.path.join(get_root_dir(), "models")
os.makedirs(MEMORY_DIR, exist_ok=True)

# --- Worker Logic ---

class MemorySaver:
    def __init__(self):
        self.current_episode_data = []
        self.episode_count = 0
        self.current_player_class = None
        self.current_model_step = 0

    def set_context(self, player_class, model_step):
        self.current_player_class = player_class
        self.current_model_step = model_step

    def save_transition(self, state, action, reward, next_state, done, reward_details, prev_game_state=None, next_game_state=None, prev_prev_game_state=None):
        # Store transition in memory buffer
        # We store CPU tensors to save space/time if they are on GPU
        if isinstance(state, torch.Tensor): state = state.cpu()
        if isinstance(next_state, torch.Tensor): next_state = next_state.cpu()
        
        # Store raw game states for potential reward recalculation
        self.current_episode_data.append({
            "state_tensor": state,
            "action": action,
            "reward": reward,
            "next_state_tensor": next_state,
            "done": done,
            "reward_details": reward_details,
            "prev_game_state": prev_game_state,
            "next_game_state": next_game_state,
            "prev_prev_game_state": prev_prev_game_state
        })

        if done:
            self.flush_episode()

    def flush_episode(self):
        if not self.current_episode_data:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Determine save directory based on player class
        save_dir = MEMORY_DIR
        class_name = "Unknown"
        if self.current_player_class:
            class_name = self.current_player_class.name
            save_dir = os.path.join(MEMORY_DIR, class_name)
            os.makedirs(save_dir, exist_ok=True)
            
        # Filename format: step_{base_step}_{game_steps}_{timestamp}.pt
        game_steps = len(self.current_episode_data)
        filename = f"step_{self.current_model_step}_{game_steps}_{timestamp}.pt"
        filepath = os.path.join(save_dir, filename)
        
        # Write to a temporary file first then rename to avoid partial reads
        temp_filepath = filepath + ".tmp"
        try:
            torch.save(self.current_episode_data, temp_filepath)
            os.rename(temp_filepath, filepath)
            sys.stderr.write(f"Saved {len(self.current_episode_data)} transitions to {filename}\n")
        except Exception as e:
            sys.stderr.write(f"Error saving memory: {e}\n")
        
        self.current_episode_data = []
        self.episode_count += 1

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

def run_worker():
    # Initialize WandB (optional, maybe for logging game stats)
    # os.environ["WANDB_SILENT"] = "true"
    # wandb_logger = WandbLogger(project_name="spire-ai-worker", run_name=f"Worker_{os.getpid()}")
    memory_saver = MemorySaver()
    
    # Initialize Agent
    # play_mode=False enables exploration (Boltzmann sampling) instead of greedy selection
    # memory_callback ensures we save data instead of training locally
    agent = DQNAgent(play_mode=False, memory_callback=memory_saver.save_transition)
    
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    player_class_cycle = itertools.cycle(PlayerClass)
    
    sys.stderr.write("Worker started. Waiting for game...\n")

    current_model_step = 0
    
    while True:
        chosen_class = next(player_class_cycle)
        model_path, step_num = get_latest_model_path()
        if model_path:
            try:
                agent.load_model(model_path)
                current_model_step = step_num
            except Exception as e:
                sys.stderr.write(f"Failed to load model: {e}\n")

        # Update memory saver context
        memory_saver.set_context(chosen_class, current_model_step)

        # 2. Play one game
        agent.change_class(chosen_class)
        
        # Play game
        coordinator.play_one_game(chosen_class, ascension_level=20)

if __name__ == "__main__":
    run_worker()
    # model_path, step = get_latest_model_path()
    # print(f"Latest model: {model_path} at step {step}")