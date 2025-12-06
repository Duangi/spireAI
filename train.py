#!/opt/miniconda3/envs/spire/bin/python3
# 使用指定的python解释器运行此脚本
import itertools
from typing import Tuple
import torch

import sys
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
import os
import time
from datetime import datetime
from spirecomm.utils.path import get_root_dir

# --- 2. 定义训练超参数 ---
NUM_EPISODES = 5000  # 总共训练5000局游戏
TARGET_UPDATE_FREQUENCY = 40 # 每 40 局游戏更新一次目标网络
SAVE_MODEL_FREQUENCY = 40 # 每 40 局游戏保存一次模型
TRAIN_BATCHES_PER_EPISODE = 64 # 每局游戏结束后，从经验池中采样训练的次数
BATCH_SIZE = 32 # 每次训练时从经验池采样的大小

# 从最新的模型开始训练
def get_latest_model_agent() -> Tuple[int,DQNAgent]:
    models_dir = os.path.join(get_root_dir(), "models")
    # 找到数字最大的模型文件
    model_files = [f for f in os.listdir(models_dir) if f.startswith("dqn_model_episode_") and f.endswith(".pth")]
    if not model_files:
        return 0, DQNAgent()
    latest_episode = 0
    latest_model_path = None
    for f in model_files:
        episode_num = int(f[len("dqn_model_episode_"):-len(".pth")])
        if episode_num > latest_episode:
            latest_episode = episode_num
            latest_model_path = os.path.join(models_dir, f)
    if latest_model_path:
        return latest_episode, DQNAgent(model_path=latest_model_path)
    else:
        return 0, DQNAgent()

def save_model_checkpoint(agent: DQNAgent, models_dir: str, episode: int, latest_episode: int):
    """
    将模型保存到 models_dir，并写入中文日志与控制台输出。
    返回 (save_path, latest_path)。
    """
    try:
        os.makedirs(models_dir, exist_ok=True)
        global_episode = episode + latest_episode if latest_episode > 0 else episode
        save_path = os.path.join(models_dir, f"dqn_model_episode_{global_episode}.pth")
        torch.save(agent.dqn_algorithm.policy_net.state_dict(), save_path)
        if latest_episode > 0:
            log_line = (f"{datetime.now().isoformat()} 已保存模型。本轮训练局数={episode}，总共训练局数={global_episode}（基于已有模型继续训练）\n")
        else:
            log_line = (f"{datetime.now().isoformat()} 已保存模型。本轮训练局数={episode}，总共训练局数={global_episode}\n")
        with open(os.path.join(models_dir, "save_model.log"), "a", encoding="utf-8") as lf:
            lf.write(log_line)
        # 另存一份最新模型
        latest_path = os.path.join(models_dir, "dqn_model_latest.pth")
        torch.save(agent.dqn_algorithm.policy_net.state_dict(), latest_path)
        return save_path, latest_path
    except Exception as e:
        return None, None

def setup_coordinator_for_agent(agent: DQNAgent) -> Coordinator:
    """
    创建 Coordinator，signal_ready 并注册 agent 的回调，返回 coordinator 实例。
    """
    coord = Coordinator()
    coord.signal_ready()
    coord.register_command_error_callback(agent.handle_error)
    coord.register_state_change_callback(agent.get_next_action_in_game)
    coord.register_out_of_game_callback(agent.get_next_action_out_of_game)
    return coord

# 单独训练某一个角色
def train_single_class(agent:DQNAgent, num_episodes:int = NUM_EPISODES, player_class:PlayerClass=PlayerClass.IRONCLAD, ascension_level: int = 20, latest_episode: int = 0):
    """
    对单个 player_class 进行训练若干局。保存时会在 models/<PLAYERCLASS.name>/ 下建立单独子目录。
    """
    if agent is None:
        _, agent = get_latest_model_agent()
    coordinator = setup_coordinator_for_agent(agent)
    chosen_class = player_class

    # 为该角色准备独立的模型保存目录：models/<PLAYERCLASS.name>
    class_models_dir = os.path.join(get_root_dir(), "models", str(player_class.name))
    # 继续使用 save_model_checkpoint(...)，它会负责创建目录

    for episode in range(1, num_episodes + 1):
        
        coordinator.play_one_game(chosen_class, ascension_level=ascension_level)

        # 学习阶段：每局结束后多次从经验池采样训练
        for _ in range(TRAIN_BATCHES_PER_EPISODE):
            agent.learn(BATCH_SIZE)

        # 周期性更新目标网络
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.dqn_algorithm.update_target_net()

        # 周期性保存模型
        if episode % SAVE_MODEL_FREQUENCY == 0:
            save_model_checkpoint(agent, class_models_dir, episode, latest_episode)

def train_all_classes(latest_episode: int = 0, num_episodes: int = NUM_EPISODES, agent: DQNAgent = None):
    """
    针对所有角色轮流训练 NUM_EPISODES 局（按常量 NUM_EPISODES）。
    会周期性更新 target_net 与保存模型（使用 SAVE_MODEL_FREQUENCY / TARGET_UPDATE_FREQUENCY）。
    """
    if agent is None:
        _, agent = get_latest_model_agent()
    coordinator = setup_coordinator_for_agent(agent)
    player_class_cycle = itertools.cycle(PlayerClass)

    for episode in range(1, num_episodes + 1):
        chosen_class = next(player_class_cycle)
        agent.change_class(chosen_class)
        coordinator.play_one_game(chosen_class, ascension_level=20)

        # 学习阶段：每局结束后多次从经验池采样训练
        for _ in range(TRAIN_BATCHES_PER_EPISODE):
            agent.learn(BATCH_SIZE)

        # 周期性更新目标网络
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            agent.dqn_algorithm.update_target_net()

        # 周期性保存模型
        if episode % SAVE_MODEL_FREQUENCY == 0:
            models_dir = os.path.join(get_root_dir(), "models")
            save_model_checkpoint(agent, models_dir, episode, latest_episode)

# --- 1. 初始化 ---
if __name__ == "__main__":
    # --- 1. 初始化 --- 
    latest_episode, dqn_agent = get_latest_model_agent()

    # 采用统一的训练入口：按需选择单角色训练或全角色训练
    # 默认行为：训练所有角色
    train_all_classes(latest_episode=latest_episode, agent=dqn_agent)