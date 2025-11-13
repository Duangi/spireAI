#!/opt/miniconda3/envs/spire/bin/python3
# 使用指定的python解释器运行此脚本
import itertools

from spirecomm.communication.coordinator import Coordinator
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.dqn import DQN,DQNAgent


if __name__ == "__main__":
    # agent = SimpleAgent()
    dqn = DQNAgent()
    card_manager = dqn.card_manager
    coordinator = Coordinator()
    coordinator.signal_ready()
    coordinator.register_command_error_callback(dqn.handle_error)
    coordinator.register_state_change_callback(dqn.get_next_action_in_game)
    coordinator.register_out_of_game_callback(dqn.get_next_action_out_of_game)

    # Play games forever, cycling through the various classes
    for chosen_class in itertools.cycle(PlayerClass):
        dqn.change_class(chosen_class)
        result = coordinator.play_one_game(chosen_class)
