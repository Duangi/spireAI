#!/opt/miniconda3/envs/spire/bin/python3
# 使用指定的python解释器运行此脚本
import itertools
import torch

import sys
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
import os
import time
from datetime import datetime
from spirecomm.utils.path import get_root_dir


if __name__ == "__main__":
    # --- 1. 初始化 ---
    # 统一使用get_root_dir获取项目根目录
    # dqn_agent = DQNAgent(os.path.join(get_root_dir(), "models", "dqn_model_episode_140.pth"))
    # dqn_agent = DQNAgent(model_path=os.path.join(get_root_dir(), "models", "dqn_model_latest.pth"))
    dqn_agent = DQNAgent()
    coordinator = Coordinator()
    
    coordinator.signal_ready()
    
    # 注册回调函数，将协调器的事件与我们的Agent方法连接起来
    coordinator.register_command_error_callback(dqn_agent.handle_error)
    coordinator.register_state_change_callback(dqn_agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(dqn_agent.get_next_action_out_of_game)

    # --- 2. 定义训练超参数 ---
    NUM_EPISODES = 5000  # 总共训练5000局游戏
    TARGET_UPDATE_FREQUENCY = 40 # 每 40 局游戏更新一次目标网络
    SAVE_MODEL_FREQUENCY = 40 # 每 40 局游戏保存一次模型
    TRAIN_BATCHES_PER_EPISODE = 64 # 每局游戏结束后，从经验池中采样训练的次数
    BATCH_SIZE = 32 # 每次训练时从经验池采样的大小

    # --- 3. 开始主训练循环 ---
    # itertools.cycle 会无限循环遍历所有角色
    player_class_cycle = itertools.cycle(PlayerClass)

    for episode in range(1, NUM_EPISODES + 1):
        chosen_class = next(player_class_cycle)
        dqn_agent.change_class(chosen_class)
        result = coordinator.play_one_game(chosen_class, ascension_level=20)

        # --- 训练网络 ---
        # 在一局游戏结束后，进行多次训练
        for _ in range(TRAIN_BATCHES_PER_EPISODE):
            dqn_agent.learn(BATCH_SIZE)

        # --- 4. 周期性任务 ---
        # 每隔N局游戏，更新目标网络
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            dqn_agent.dqn_algorithm.update_target_net()

        # 每隔N局游戏，保存模型权重
        if episode % SAVE_MODEL_FREQUENCY == 0:
            # 明确保存到项目下的 models 目录，便于查找
            
            models_dir = os.path.join(get_root_dir(), "models")
            try:
                os.makedirs(models_dir, exist_ok=True)
                save_path = os.path.join(models_dir, f"dqn_model_episode_{episode}.pth")
                torch.save(dqn_agent.dqn_algorithm.policy_net.state_dict(), save_path)
                # 记录一次保存事件到日志，帮助排查
                with open(os.path.join(models_dir, "save_model.log"), "a", encoding="utf-8") as lf:
                    lf.write(f"{datetime.now().isoformat()} Saved model for episode {episode} -> {save_path}\n")
                # 存一个最新的模型副本，方便快速加载
                latest_path = os.path.join(models_dir, "dqn_model_latest.pth")
                torch.save(dqn_agent.dqn_algorithm.policy_net.state_dict(), latest_path)
            except Exception as e:
                # 如果写入失败，把异常记录下来但不抛出，避免影响训练循环
                err_msg = f"{datetime.now().isoformat()} Failed to save model for episode {episode}: {e}\n"
                try:
                    with open(os.path.join(models_dir, "save_model.log"), "a", encoding="utf-8") as lf:
                        lf.write(err_msg)
                except Exception:
                    pass
