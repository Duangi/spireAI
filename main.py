#!/opt/miniconda3/envs/spire/bin/python3
# 使用指定的python解释器运行此脚本
import itertools
import torch

import sys
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent


if __name__ == "__main__":
    # --- 游玩模式配置 ---
    # 指定你要加载的模型文件路径
    # 将其替换为你训练好的模型，例如 "dqn_model_episode_5000.pth"
    MODEL_PATH = "dqn_model_episode_100.pth" 

    # --- 1. 初始化 ---
    # 以 play_mode=True 初始化 Agent，并传入模型路径
    dqn_agent = DQNAgent(play_mode=True, model_path=MODEL_PATH)
    coordinator = Coordinator()
    coordinator.signal_ready()
    
    # 注册回调函数
    coordinator.register_command_error_callback(dqn_agent.handle_error)
    coordinator.register_state_change_callback(dqn_agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(dqn_agent.get_next_action_out_of_game)

    # --- 2. 开始无限游玩 ---
    for chosen_class in itertools.cycle(PlayerClass):
        print(f"\n--- Starting new game with {chosen_class.name} ---", file=sys.stderr)
        dqn_agent.change_class(chosen_class)
        result = coordinator.play_one_game(chosen_class)
        print(f"--- Game Finished. Result: {'Victory' if result else 'Defeat'} ---", file=sys.stderr)
