#!/opt/miniconda3/envs/spire/bin/python3
# 使用指定的python解释器运行此脚本
import os
# --- Proxy Configuration ---
# Explicitly set proxy for this process to ensure WandB connectivity
# This avoids affecting global system settings or VS Code extensions
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
# ---------------------------



# Silence WandB console output
os.environ["WANDB_SILENT"] = "true"
# --------------------------

import itertools
from typing import Tuple
import torch

import sys
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
from spirecomm.ai.dqn_core.model import SpireConfig
import os
import time
from datetime import datetime
from spirecomm.utils.path import get_root_dir

# --- 2. 定义训练超参数 ---
MAX_STEPS = 2000000  # 总共训练 200w steps
TARGET_UPDATE_STEPS = 2000 # 每 2000 steps 更新一次目标网络
SAVE_MODEL_STEPS = 5000 # 每 5000 steps 保存一次模型
TRAIN_BATCHES_PER_EPISODE = 64 # 每局游戏结束后，从经验池中采样训练的次数
BATCH_SIZE = 32 # 每次训练时从经验池采样的大小

# 从最新的模型开始训练
def get_latest_model_agent(player_class: PlayerClass = None, wandb_logger: WandbLogger = None) -> Tuple[int,DQNAgent]:
    models_dir = os.path.join(get_root_dir(), "models")
    if player_class:
        models_dir = os.path.join(models_dir, player_class.name)
    # 找到数字最大的模型文件,如果没有则返回0和新建的DQNAgent
    if not os.path.exists(models_dir):
        return 0, DQNAgent(wandb_logger=wandb_logger)
    
    # 修改：查找 step_xxx.pth
    model_files = [f for f in os.listdir(models_dir) if f.startswith("step_") and f.endswith(".pth")]
    if not model_files:
        return 0, DQNAgent(wandb_logger=wandb_logger)
    
    latest_step = 0
    latest_model_path = None
    for f in model_files:
        try:
            step_num = int(f[len("step_"):-len(".pth")])
            if step_num > latest_step:
                latest_step = step_num
                latest_model_path = os.path.join(models_dir, f)
        except ValueError:
            continue
            
    if latest_model_path:
        agent = DQNAgent(model_path=latest_model_path, wandb_logger=wandb_logger)
        # 假设 agent 有 steps_done 属性，需要同步为加载的 step
        if hasattr(agent, 'steps_done'):
            agent.steps_done = latest_step
        return latest_step, agent
    else:
        return 0, DQNAgent(wandb_logger=wandb_logger)

def save_model_checkpoint(agent: DQNAgent, models_dir: str, current_step: int):
    """
    将模型保存到 models_dir，并写入中文日志与控制台输出。
    保存为 step_xxx.pth 和 latest.pth
    """
    try:
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, f"step_{current_step}.pth")
        torch.save(agent.dqn_algorithm.policy_net.state_dict(), save_path)
        
        log_line = (f"{datetime.now().isoformat()} 已保存模型。当前Steps={current_step}\n")
        with open(os.path.join(models_dir, "save_model.log"), "a", encoding="utf-8") as lf:
            lf.write(log_line)
            
        # 另存一份最新模型
        latest_path = os.path.join(models_dir, "latest.pth")
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
def train_single_class(agent:DQNAgent, max_steps:int = MAX_STEPS, player_class:PlayerClass=PlayerClass.IRONCLAD, ascension_level: int = 20):
    """
    对单个 player_class 进行训练。保存时会在 models/<PLAYERCLASS.name>/ 下建立单独子目录。
    """
    if agent is None:
        _, agent = get_latest_model_agent()
    coordinator = setup_coordinator_for_agent(agent)
    
    # 注册退出回调，确保 WandB 正确结束
    def on_exit():
        if agent.dqn_algorithm.wandb_logger:
            agent.dqn_algorithm.wandb_logger.finish()
    coordinator.register_on_exit_callback(on_exit)

    chosen_class = player_class
    agent.change_class(chosen_class)
    # 为该角色准备独立的模型保存目录：models/<PLAYERCLASS.name>
    class_models_dir = os.path.join(get_root_dir(), "models", str(player_class.name))
    
    # 记录上一次保存和更新的 step
    last_save_step = agent.steps_done if hasattr(agent, 'steps_done') else 0
    last_target_update_step = last_save_step

    while True:
        current_steps = agent.steps_done if hasattr(agent, 'steps_done') else 0
        if current_steps >= max_steps:
            break

        coordinator.play_one_game(chosen_class, ascension_level=ascension_level)

        # 学习阶段：每局结束后多次从经验池采样训练
        for _ in range(TRAIN_BATCHES_PER_EPISODE):
            agent.learn(BATCH_SIZE)

        # 获取最新 step
        current_steps = agent.steps_done if hasattr(agent, 'steps_done') else 0

        # 周期性更新目标网络
        if current_steps - last_target_update_step >= TARGET_UPDATE_STEPS:
            agent.dqn_algorithm.update_target_net()
            last_target_update_step = current_steps

        # 周期性保存模型
        if current_steps - last_save_step >= SAVE_MODEL_STEPS:
            save_model_checkpoint(agent, class_models_dir, current_steps)
            last_save_step = current_steps

def train_all_classes(agent: DQNAgent = None, max_steps: int = MAX_STEPS, ascension_level: int = 20):
    """
    针对所有角色轮流训练，直到达到 max_steps。
    """
    if agent is None:
        _, agent = get_latest_model_agent()
    coordinator = setup_coordinator_for_agent(agent)
    
    # 注册退出回调，确保 WandB 正确结束
    def on_exit():
        if agent.dqn_algorithm.wandb_logger:
            agent.dqn_algorithm.wandb_logger.finish()
    coordinator.register_on_exit_callback(on_exit)

    player_class_cycle = itertools.cycle(PlayerClass)
    
    # 记录上一次保存和更新的 step
    last_save_step = agent.steps_done if hasattr(agent, 'steps_done') else 0
    last_target_update_step = last_save_step

    while True:
        current_steps = agent.steps_done if hasattr(agent, 'steps_done') else 0
        if current_steps >= max_steps:
            break

        chosen_class = next(player_class_cycle)
        agent.change_class(chosen_class)
        coordinator.play_one_game(chosen_class, ascension_level=ascension_level)

        # 学习阶段：每局结束后多次从经验池采样训练
        for _ in range(TRAIN_BATCHES_PER_EPISODE):
            agent.learn(BATCH_SIZE)

        # 获取最新 step
        current_steps = agent.steps_done if hasattr(agent, 'steps_done') else 0

        # 周期性更新目标网络
        if current_steps - last_target_update_step >= TARGET_UPDATE_STEPS:
            agent.dqn_algorithm.update_target_net()
            last_target_update_step = current_steps

        # 周期性保存模型
        if current_steps - last_save_step >= SAVE_MODEL_STEPS:
            models_dir = os.path.join(get_root_dir(), "models")
            save_model_checkpoint(agent, models_dir, current_steps)
            last_save_step = current_steps

# --- 1. 初始化 ---
if __name__ == "__main__":
    # 初始化 WandbLogger
    wandb_logger = WandbLogger(project_name="spire-ai-train", run_name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    # 在这里修改需要训练的角色与参数
    player_class_to_train = PlayerClass.THE_SILENT
    train_single_class_mode = False
    ascension_level_to_train = 20
    
    # 获取最新模型 (注意：这里返回的是 latest_step)
    latest_step, dqn_agent = get_latest_model_agent(player_class_to_train if train_single_class_mode else None, wandb_logger=wandb_logger)

    # 采用统一的训练入口：按需选择单角色训练或全角色训练
    # 默认行为：训练所有角色
    if train_single_class_mode:
        train_single_class(dqn_agent, max_steps=MAX_STEPS, ascension_level=ascension_level_to_train, player_class=player_class_to_train)
    else:
        train_all_classes(dqn_agent, max_steps=MAX_STEPS, ascension_level=ascension_level_to_train)