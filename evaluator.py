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
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)


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


class MemorySaver:
    """与 worker.py 相同的 episodic 存储逻辑，用于生成可训练的数据文件。"""
    def __init__(self):
        self.current_episode_data = []
        self.episode_count = 0
        self.current_player_class = None
        self.current_model_step = 0

    def set_context(self, player_class, model_step):
        self.current_player_class = player_class
        self.current_model_step = model_step

    def save_transition(self, state, action, reward, next_state, done, reward_details,
                        prev_game_state=None, next_game_state=None, prev_prev_game_state=None):
        # 保证张量在 CPU 上存储，节省显存
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()

        self.current_episode_data.append({
            "state_tensor": state,
            "action": action,
            "reward": reward,
            "next_state_tensor": next_state,
            "done": done,
            "reward_details": reward_details,
            "prev_game_state": prev_game_state,
            "next_game_state": next_game_state,
            "prev_prev_game_state": prev_prev_game_state,
        })

        if done:
            self.flush_episode()

    def flush_episode(self):
        if not self.current_episode_data:
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        save_dir = MEMORY_DIR
        class_name = "Unknown"
        if self.current_player_class:
            class_name = self.current_player_class.name
            save_dir = os.path.join(MEMORY_DIR, class_name)
            os.makedirs(save_dir, exist_ok=True)

        game_steps = len(self.current_episode_data)
        filename = f"step_{self.current_model_step}_{game_steps}_{timestamp}.pt"
        filepath = os.path.join(save_dir, filename)
        temp_filepath = filepath + ".tmp"
        try:
            torch.save(self.current_episode_data, temp_filepath)
            os.rename(temp_filepath, filepath)
            sys.stderr.write(f"[Evaluator] Saved {len(self.current_episode_data)} transitions to {filename}\n")
        except Exception as e:
            sys.stderr.write(f"[Evaluator] Error saving memory: {e}\n")

        self.current_episode_data = []
        self.episode_count += 1


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
    """评估者模式：使用推理模式模型打游戏，但仍通过 memory_callback 生成训练数据文件。"""
    # Allow class via CLI arg or interactive
    arg = sys.argv[1] if len(sys.argv) > 1 else None
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

    print(f"[Evaluator] Selected class: {chosen_class.name}")

    # 初始化 MemorySaver，用于写入 data/memory 结构
    memory_saver = MemorySaver()

    # Initialize agent in evaluation mode:
    # - play_mode=True: 使用推理策略（非探索）
    # - memory_callback: 仍然把经验写入文件，供 trainer 使用
    agent = DQNAgent(play_mode=True, memory_callback=memory_saver.save_transition)
    try:
        agent.dqn_algorithm.set_inference_mode()
        agent.dqn_algorithm.policy_net.eval()
    except Exception:
        pass

    # Load latest model for this class if available
    model_path, step = get_latest_model_path(chosen_class)
    current_model_step = step or 0
    if model_path:
        try:
            print(f"[Evaluator] Loading model {model_path} (step {step})...")
            agent.load_model(model_path)
        except Exception as e:
            print(f"[Evaluator] Failed to load model: {e}")
    else:
        print("[Evaluator] No model found, using random-initialized weights.")

    # 更新 MemorySaver 的上下文，使保存的文件名包含当前模型 step
    memory_saver.set_context(chosen_class, current_model_step)

    # Coordinator setup
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
    coordinator.register_command_error_callback(agent.handle_error)

    print("[Evaluator] Starting evaluation playthrough...")
    coordinator.play_one_game(chosen_class, ascension_level=0)
    print("[Evaluator] Evaluation finished.")


if __name__ == '__main__':
    main()
