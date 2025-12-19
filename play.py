import os
import sys
import time
import glob
import shutil
import torch
from datetime import datetime

from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.utils.path import get_root_dir

# --- Configuration ---
MODELS_DIR = os.path.join(get_root_dir(), "models")


def get_latest_model_path(player_class=None):
    target_dir = MODELS_DIR
    if player_class:
        class_dir = os.path.join(MODELS_DIR, player_class.name)
        if os.path.exists(class_dir):
            target_dir = class_dir

    if not os.path.exists(target_dir):
        return None, 0

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


def choose_player_class_interactive():
    classes = list(PlayerClass)
    print("Available classes:")
    for i, c in enumerate(classes):
        print(f"  {i}: {c.name}")
    try:
        sel = input("Select class index (or press Enter for 0): ")
    except Exception:
        sel = ''
    if sel.strip() == '':
        return classes[0]
    try:
        idx = int(sel)
        return classes[idx]
    except Exception:
        # try by name
        try:
            return PlayerClass[sel]
        except Exception:
            print("Invalid selection, defaulting to first class.")
            return classes[0]


def main():
    # Allow class via CLI arg or interactive
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    chosen_class = None
    if arg:
        try:
            chosen_class = PlayerClass[arg]
        except Exception:
            try:
                chosen_class = PlayerClass[int(arg)]
            except Exception:
                print(f"Unknown class '{arg}', falling back to interactive selection.")
                chosen_class = choose_player_class_interactive()
    else:
        chosen_class = choose_player_class_interactive()

    print(f"Selected class: {chosen_class.name}")

    # Initialize agent in play mode
    agent = DQNAgent(play_mode=True)
    # Ensure inference mode
    try:
        agent.dqn_algorithm.set_inference_mode()
        agent.dqn_algorithm.policy_net.eval()
    except Exception:
        pass

    # Load latest model for this class if available
    model_path, step = get_latest_model_path(chosen_class)
    if model_path:
        try:
            print(f"Loading model {model_path} (step {step})...")
            agent.load_model(model_path)
        except Exception as e:
            print(f"Failed to load model: {e}")

    # Coordinator setup
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    coordinator.register_command_error_callback(agent.handle_error)

    print("Starting one playthrough...")
    coordinator.play_one_game(chosen_class, ascension_level=0)
    print("Play finished.")


if __name__ == '__main__':
    main()
