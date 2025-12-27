import os
import sys
import time
from datetime import datetime

import torch

from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.game import Game
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.utils.path import get_root_dir

# 和 worker/evaluator 一致的存储目录
MEMORY_DIR = os.path.join(get_root_dir(), "data", "memory")
os.makedirs(MEMORY_DIR, exist_ok=True)


class HumanMemorySaver:
    """仿照 worker 的 MemorySaver，把整局人类操作的轨迹写入 data/memory 供训练使用。"""

    def __init__(self):
        self.current_episode_data = []
        self.episode_count = 0
        self.current_player_class = None
        self.current_model_step = 0  # 人类局面，不依赖模型，标记为 0

    def set_context(self, player_class, model_step: int = 0):
        self.current_player_class = player_class
        self.current_model_step = model_step

    def save_transition(self, state_tensor, action, reward, next_state_tensor, done,
                        reward_details=None, prev_game_state=None, next_game_state=None,
                        prev_prev_game_state=None):
        # 与 worker 一致：确保存到 CPU
        if isinstance(state_tensor, torch.Tensor):
            state_tensor = state_tensor.cpu()
        if isinstance(next_state_tensor, torch.Tensor):
            next_state_tensor = next_state_tensor.cpu()

        self.current_episode_data.append({
            "state_tensor": state_tensor,
            "action": action,
            "reward": reward,
            "next_state_tensor": next_state_tensor,
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
        if self.current_player_class is not None:
            save_dir = os.path.join(MEMORY_DIR, self.current_player_class.name)
            os.makedirs(save_dir, exist_ok=True)

        game_steps = len(self.current_episode_data)
        filename = f"step_{self.current_model_step}_{game_steps}_{timestamp}.pt"
        filepath = os.path.join(save_dir, filename)
        temp_filepath = filepath + ".tmp"

        try:
            torch.save(self.current_episode_data, temp_filepath)
            os.rename(temp_filepath, filepath)
        except Exception:
            pass

        self.current_episode_data = []
        self.episode_count += 1


class HumanAgent:
    """不做自动决策，仅记录人类在游戏界面中做出的操作。"""

    def __init__(self, memory_saver: HumanMemorySaver):
        self.memory_saver = memory_saver
        self.state_processor = GameStateProcessor()
        self.last_state_tensor = None
        self.last_game_state = None
        self.last_action_cmd = None

    def get_next_action_in_game(self, game: Game):
        """在战斗/地图等游戏状态下被 Coordinator 回调。"""
        # 1. 构造当前 state tensor（在 CPU 上即可，后面训练时再搬到 GPU）
        state_tensor = self.state_processor.get_state_tensor(game)

        # 从游戏对象中拿到上一帧已经执行完的人类指令
        available = getattr(game, "available_commands", None) or []
        # 默认假设游戏已经按人类点击选择了第 0 个 available_commands
        action_cmd = available[0] if available else None

        # 2. 保存上一步 transition（如果有上一步）
        if self.last_state_tensor is not None and self.last_game_state is not None:
            reward = 0.0
            done = False
            self.memory_saver.save_transition(
                self.last_state_tensor,
                self.last_action_cmd,
                reward,
                state_tensor,
                done,
                reward_details=None,
                prev_game_state=self.last_game_state,
                next_game_state=game,
            )

        # 3. 更新 last_*，返回当前动作命令
        self.last_state_tensor = state_tensor
        self.last_game_state = game
        self.last_action_cmd = action_cmd

        return action_cmd

    def get_next_action_out_of_game(self, game: Game):
        """非战斗/菜单等状态下的回调，逻辑同上。"""
        return self.get_next_action_in_game(game)

    def handle_error(self, message: str):
        # 静默处理，不打印
        pass

    def end_episode(self, final_reward: float = 0.0):
        """在一局游戏结束时调用，给最后一步打上 done=True 并 flush。"""
        if self.last_state_tensor is not None and self.last_game_state is not None:
            self.memory_saver.save_transition(
                self.last_state_tensor,
                self.last_action_cmd,
                final_reward,
                self.last_state_tensor,
                True,
                reward_details={"final": True},
                prev_game_state=self.last_game_state,
                next_game_state=None,
            )
        self.last_state_tensor = None
        self.last_game_state = None
        self.last_action_cmd = None


def main():
    # 不使用参数和交互，按职业列表循环开局
    classes = list(PlayerClass)

    memory_saver = HumanMemorySaver()
    agent = HumanAgent(memory_saver)

    while True:
        for chosen_class in classes:
            memory_saver.set_context(chosen_class, model_step=0)

            coordinator = Coordinator()
            coordinator.signal_ready()
            coordinator.register_state_change_callback(agent.get_next_action_in_game)
            coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)
            coordinator.register_command_error_callback(agent.handle_error)

            coordinator.play_one_game(chosen_class, ascension_level=0)

            agent.end_episode(final_reward=0.0)


if __name__ == '__main__':
    main()
