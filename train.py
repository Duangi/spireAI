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
if __name__ == "__main__":
    # --- 1. 初始化 ---
    latest_episode, dqn_agent = get_latest_model_agent()
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
                # 全局序号（兼容从已有最新模型继续训练的情况）
                global_episode = episode + latest_episode if latest_episode > 0 else episode
                save_path = os.path.join(models_dir, f"dqn_model_episode_{global_episode}.pth")
                # 保存指定 episode 的模型
                torch.save(dqn_agent.dqn_algorithm.policy_net.state_dict(), save_path)
                # 记录中文详细日志（含本轮局号与全局序号）
                with open(os.path.join(models_dir, "save_model.log"), "a", encoding="utf-8") as lf:
                    if latest_episode > 0:
                        lf.write(f"{datetime.now().isoformat()} 已保存模型。当前模型从 {episode} 局训练之后继续训练。训练完成之后的总轮数为：{global_episode}）。\n")
                    lf.write(f"{datetime.now().isoformat()} 已保存模型。目前总训练局数为：{episode}。\n")
                # 另外存一个最新模型副本，便于快速加载
                latest_path = os.path.join(models_dir, "dqn_model_latest.pth")
                torch.save(dqn_agent.dqn_algorithm.policy_net.state_dict(), latest_path)
            except Exception as e:
                print(f"保存模型时出错: {e}", file=sys.stderr)