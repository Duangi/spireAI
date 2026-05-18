import os
import sys
import torch
import itertools
import time
import random
from datetime import datetime
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.reward_scheduler import RewardAutoScheduler
from spirecomm.utils.path import get_root_dir

# --- Configuration ---
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
MODELS_DIR = os.path.join(get_root_dir(), "models")
os.makedirs(MEMORY_DIR, exist_ok=True)

# Detect device for this worker (prefer CUDA)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        save_dir = MEMORY_DIR
        if self.current_player_class:
            save_dir = os.path.join(MEMORY_DIR, self.current_player_class.name)
            os.makedirs(save_dir, exist_ok=True)
            
        game_steps = len(self.current_episode_data)
        filename = f"step_{self.current_model_step}_{game_steps}_{timestamp}.pt"
        filepath = os.path.join(save_dir, filename)
        
        temp_filepath = filepath + ".tmp"
        try:
            torch.save(self.current_episode_data, temp_filepath)
            os.rename(temp_filepath, filepath)
            sys.stderr.write(f"[Worker] Saved {game_steps} transitions to {filename}\n")
        except Exception as e:
            sys.stderr.write(f"Error saving memory: {e}\n")
        
        self.current_episode_data = []
        self.episode_count += 1

def get_latest_model_path():
    """仅用于获取当前最新模型的步数编号"""
    target_dir = MODELS_DIR
    model_files = [f for f in os.listdir(target_dir) if f.startswith("step_") and f.endswith(".pth")]
    latest_step = 0
    latest_path = None
    for f in model_files:
        try:
            step_num = int(f[len("step_"):-len(".pth")])
            if step_num > latest_step:
                latest_step = step_num
                latest_path = os.path.join(target_dir, f)
        except ValueError:
            continue
    return latest_path, latest_step

def run_worker():
    memory_saver = MemorySaver()
    agent = DQNAgent(play_mode=False, memory_callback=memory_saver.save_transition)
    reward_scheduler = RewardAutoScheduler()
    scheduler_state = reward_scheduler.initialize()
    sys.stderr.write(
        "[Worker] Reward auto scheduler initialized: "
        f"stage={scheduler_state.next_stage} avg_floor={scheduler_state.metrics['avg_floor']:.2f}\n"
    )

    # 移动模型到正确设备（初始）
    try:
        if hasattr(agent.dqn_algorithm, "policy_net"):
            agent.dqn_algorithm.policy_net.to(DEVICE)
    except Exception as e:
        sys.stderr.write(f"[WARN] Initial device move failed: {e}\n")
    
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(agent.handle_error)
    coordinator.register_state_change_callback(agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

    player_class_cycle = itertools.cycle(PlayerClass)
    
    # 状态跟踪变量
    last_loaded_mtime = 0
    current_model_step = 0
    latest_model_file = os.path.join(MODELS_DIR, "latest.pth")

    sys.stderr.write(f"Worker initialized on {DEVICE}. Entering main loop...\n")

    while True:
        chosen_class = next(player_class_cycle)
        
        # 1. 按需加载模型
        if os.path.exists(latest_model_file):
            try:
                mtime = os.path.getmtime(latest_model_file)
                if mtime > last_loaded_mtime:
                    # 发现新模型，执行加载
                    sys.stderr.write(f"[Worker] Loading updated latest.pth...\n")
                    agent.load_model(latest_model_file)
                    
                    # 仅在模型更新时通过扫描文件夹获取一次当前 step 数，用于存数据时的命名
                    _, step_num = get_latest_model_path()
                    current_model_step = step_num
                    
                    # 确保权重移动到正确设备
                    if hasattr(agent.dqn_algorithm, "policy_net"):
                        agent.dqn_algorithm.policy_net.to(DEVICE)
                    
                    last_loaded_mtime = mtime
                    sys.stderr.write(f"[Worker] Model updated to step {current_model_step}\n")
                else:
                    # 模型未变，跳过加载逻辑
                    pass
            except Exception as e:
                sys.stderr.write(f"[Error] Failed to load latest model: {e}\n")

        # 2. 按需加载配置（如果有动态配置文件）
        agent.reward_calculator.reload_config()
        
        # 3. 更新上下文并重置 Agent 状态
        memory_saver.set_context(chosen_class, current_model_step)
        agent.change_class(chosen_class)
        
        # 4. 随机微小延迟，错开多个 Worker 的启动峰值
        time.sleep(random.uniform(0.1, 1.0))

        # 5. 执行一局游戏
        try:
            coordinator.play_one_game(chosen_class, ascension_level=1)
            final_floor = 0
            victory = False
            if coordinator.last_game_state is not None:
                final_floor = int(getattr(coordinator.last_game_state, "floor", 0) or 0)
                screen = getattr(coordinator.last_game_state, "screen", None)
                victory = bool(getattr(screen, "victory", False))
            scheduler_update = reward_scheduler.record_episode(
                floor_reached=final_floor,
                victory=victory,
                player_class=getattr(chosen_class, "name", str(chosen_class)),
            )
            if scheduler_update.changed:
                sys.stderr.write(
                    "[Worker] Reward auto scheduler promoted "
                    f"stage {scheduler_update.current_stage} -> {scheduler_update.next_stage} "
                    f"(avg_floor={scheduler_update.metrics['avg_floor']:.2f}, "
                    f"act2_rate={scheduler_update.metrics['act2_reach_rate']:.2%}, "
                    f"act3_rate={scheduler_update.metrics['act3_reach_rate']:.2%}, "
                    f"victory_rate={scheduler_update.metrics['victory_rate']:.2%})\n"
                )
            else:
                sys.stderr.write(
                    "[Worker] Reward auto scheduler kept "
                    f"stage={scheduler_update.next_stage} "
                    f"(floor={final_floor}, avg_floor={scheduler_update.metrics['avg_floor']:.2f})\n"
                )
        except Exception as e:
            sys.stderr.write(f"[Runtime Error] Game session crashed: {e}\n")
            time.sleep(2) # 发生异常等两秒再重开

if __name__ == "__main__":
    run_worker()
