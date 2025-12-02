#!/opt/miniconda3/envs/spire/bin/python3
# 使用指定的python解释器运行此脚本
import itertools
from numpy import absolute
import torch
import threading
import queue
import time
import os

import sys
from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQNAgent
from spirecomm.ai.absolute_logger import AbsoluteLogger, LogType

if __name__ == "__main__":
    # --- 1. 初始化 ---
    dqn_agent = DQNAgent()
    coordinator = Coordinator()
    coordinator.signal_ready()
    absolute_logger = AbsoluteLogger(LogType.TRAIN)
    absolute_logger.start_episode()
    
    # 注册回调函数，将协调器的事件与我们的Agent方法连接起来
    coordinator.register_command_error_callback(dqn_agent.handle_error)
    coordinator.register_state_change_callback(dqn_agent.get_next_action_in_game)
    coordinator.register_out_of_game_callback(dqn_agent.get_next_action_out_of_game)

    # 超时与重试配置（可通过环境变量调整）
    GAME_CALL_TIMEOUT = int(os.environ.get("GAME_CALL_TIMEOUT_SEC", "300"))  # 默认 300s
    GAME_CALL_RETRY_ON_TIMEOUT = int(os.environ.get("GAME_CALL_RETRY_ON_TIMEOUT", "1"))  # 超时后重试次数

    # helper：在单独线程中运行 coordinator.play_one_game 并通过 queue 返回结果/异常
    def run_play_one_game_with_timeout(coord, player_cls, timeout):
        q = queue.Queue()

        def target():
            try:
                res = coord.play_one_game(player_cls)
                q.put(("ok", res))
            except Exception as e:
                q.put(("err", e))

        t = threading.Thread(target=target, daemon=True)
        t.start()
        try:
            status, payload = q.get(timeout=timeout)
            return status, payload
        except queue.Empty:
            return "timeout", None

    # --- 2. 定义训练超参数 ---
    NUM_EPISODES = 5000  # 总共训练5000局游戏
    TARGET_UPDATE_FREQUENCY = 10 # 每 10 局游戏更新一次目标网络
    SAVE_MODEL_FREQUENCY = 100 # 每 100 局游戏保存一次模型
    TRAIN_BATCHES_PER_EPISODE = 64 # 每局游戏结束后，从经验池中采样训练的次数
    BATCH_SIZE = 32 # 每次训练时从经验池采样的大小

    # --- 3. 开始主训练循环 ---
    # itertools.cycle 会无限循环遍历所有角色
    player_class_cycle = itertools.cycle(PlayerClass)

    for episode in range(1, NUM_EPISODES + 1):
        chosen_class = next(player_class_cycle)
        absolute_logger.write(f"--- Starting Episode {episode}/{NUM_EPISODES}, Class: {chosen_class.name} ---")
        dqn_agent.change_class(chosen_class)

        # 在调用 play_one_game 之前再确认一次已发出 ready 信号（防护）
        try:
            absolute_logger.write("Sending initial ready signal to coordinator...")
            coordinator.signal_ready()
        except Exception as e:
            absolute_logger.write(f"signal_ready() raised exception: {e}")

        # 用超时保护调用 play_one_game，并在超时后重试（可重试多次）
        result = None
        for attempt in range(GAME_CALL_RETRY_ON_TIMEOUT + 1):
            absolute_logger.write(f"Calling play_one_game (attempt {attempt+1}/{GAME_CALL_RETRY_ON_TIMEOUT+1}) with timeout {GAME_CALL_TIMEOUT}s...")
            status, payload = run_play_one_game_with_timeout(coordinator, chosen_class, GAME_CALL_TIMEOUT)
            if status == "ok":
                result = payload
                absolute_logger.write(f"play_one_game returned successfully: {result}")
                break
            elif status == "err":
                absolute_logger.write(f"play_one_game raised exception: {payload}")
                # 发生异常时不重试（除非上层需要），记录并跳出或继续根据需要
                break
            else:  # timeout
                absolute_logger.write(f"play_one_game timed out after {GAME_CALL_TIMEOUT}s on attempt {attempt+1}.")
                # 重新发送 ready，等待对方响应；小延迟后重试
                try:
                    absolute_logger.write("Re-sending coordinator.signal_ready() after timeout...")
                    coordinator.signal_ready()
                except Exception as e:
                    absolute_logger.write(f"signal_ready() during retry raised exception: {e}")
                # 小睡以给对端一些时间
                time.sleep(2)

        if result is None:
            absolute_logger.write(f"Episode {episode}: No result from play_one_game (timeout or error). Proceeding to next episode or training step.")
        else:
            absolute_logger.write(f"--- Episode {episode} Finished. Result: {result} ---")

        # --- 训练网络 ---
        absolute_logger.write(f"Training network for {TRAIN_BATCHES_PER_EPISODE} batches...")
        for _ in range(TRAIN_BATCHES_PER_EPISODE):
            dqn_agent.learn(BATCH_SIZE)

        # --- 4. 周期性任务 ---
        # 每隔N局游戏，更新目标网络
        if episode % TARGET_UPDATE_FREQUENCY == 0:
            absolute_logger.write(f"Updating target network at episode {episode}")
            dqn_agent.dqn_algorithm.update_target_net()

        # 每隔N局游戏，保存模型权重
        if episode % SAVE_MODEL_FREQUENCY == 0:
            absolute_logger.write(f"Saving model at episode {episode}")
            torch.save(dqn_agent.dqn_algorithm.policy_net.state_dict(), f"dqn_model_episode_{episode}.pth")