import sys
import queue
import threading
import json
import collections
from typing import Optional
import os

from sympy import Abs


from spirecomm.ai.absolute_logger import AbsoluteLogger
from spirecomm.spire.game import Game
from spirecomm.spire.screen import ScreenType
from spirecomm.communication.action import Action, StartGameAction
from spirecomm.spire.character import PlayerClass
from spirecomm.utils.path import get_root_dir

def read_stdin(input_queue):
    """Read lines from stdin and write them to a queue

    :param input_queue: A queue, to which lines from stdin will be written
    :type input_queue: queue.Queue
    :return: None
    """
    while True:
        stdin_input = ""
        while True:
            input_char = sys.stdin.read(1)
            if input_char == '\n':
                break
            else:
                stdin_input += input_char
        # 输入到txt文件
        with open('process.txt', 'a') as f:
            f.write('-----input-----' + '\n')
            f.write(stdin_input + '\n')
        input_queue.put(stdin_input)


def write_stdout(output_queue):
    """Read lines from a queue and write them to stdout

    :param output_queue: A queue, from which this function will receive lines of text
    :type output_queue: queue.Queue
    :return: None
    """
    while True:
        output = output_queue.get()
        # 输出到txt文件
        with open('process.txt', 'a') as f:
            f.write('-----output-----' + '\n')
            f.write(output + '\n')
        print(output, end='\n', flush=True)


class Coordinator:
    """An object to coordinate communication with Slay the Spire"""

    def __init__(self) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.input_thread = threading.Thread(target=read_stdin, args=(self.input_queue,))
        self.output_thread = threading.Thread(target=write_stdout, args=(self.output_queue,))
        self.input_thread.daemon = True
        self.input_thread.start()
        self.output_thread.daemon = True
        self.output_thread.start()
        self.action_queue = collections.deque()
        self.state_change_callback = None
        self.out_of_game_callback = None
        self.error_callback = None
        self.game_is_ready = False
        self.stop_after_run = False
        self.in_game = False
        self.last_game_state = None
        self.last_error = None

        self.absolute_logger = AbsoluteLogger()
        self.absolute_logger.start_episode()

    def signal_ready(self):
        """Indicate to Communication Mod that setup is complete

        Must be used once, before any other commands can be sent.
        :return: None
        """
        self.send_message("ready")

    def send_message(self, message):
        """Send a command to Communication Mod and start waiting for a response

        :param message: the message to send
        :type message: str
        :return: None
        """
        # 在此处添加日志记录，记录所有即将发送的指令
        with open('sent_commands.txt', 'a', encoding='utf-8') as f:
            f.write(message + '\n')

        self.output_queue.put(message)
        self.game_is_ready = False

    def add_action_to_queue(self, action):
        """Queue an action to perform when ready

        :param action: the action to queue
        :type action: Action
        :return: None
        """
        self.action_queue.append(action)

    def clear_actions(self):
        """Remove all actions from the action queue

        :return: None
        """
        self.action_queue.clear()

    def execute_next_action(self):
        """Immediately execute the next action in the action queue

        :return: None
        """
        action = self.action_queue.popleft()
        action.execute(self)

    def execute_next_action_if_ready(self):
        """Immediately execute the next action in the action queue, if ready to do so

        :return: None
        """
        if len(self.action_queue) > 0 and self.action_queue[0].can_be_executed(self):
            self.execute_next_action()

    def register_state_change_callback(self, new_callback):
        """Register a function to be called when a message is received from Communication Mod

        :param new_callback: the function to call
        :type new_callback: function(game_state: Game) -> Action
        :return: None
        """
        self.state_change_callback = new_callback

    def register_command_error_callback(self, new_callback):
        """Register a function to be called when an error is received from Communication Mod

        :param new_callback: the function to call
        :type new_callback: function(error: str) -> Action
        :return: None
        """
        self.error_callback = new_callback

    def register_out_of_game_callback(self, new_callback):
        """Register a function to be called when Communication Mod indicates we are in the main menu

        :param new_callback: the function to call
        :type new_callback: function() -> Action
        :return: None
        """
        self.out_of_game_callback = new_callback

    def get_next_raw_message(self, block=False):
        """Get the next message from Communication Mod as a string

        :param block: set to True to wait for the next message
        :type block: bool
        :return: the message from Communication Mod
        :rtype: str
        """
        if block or not self.input_queue.empty():
            return self.input_queue.get()

    def receive_game_state_update(self, block=False, perform_callbacks=True):
        """Using the next message from Communication Mod, update the stored game state

        :param block: set to True to wait for the next message
        :type block: bool
        :param perform_callbacks: set to True to perform callbacks based on the new game state
        :type perform_callbacks: bool
        :return: whether a message was received
        """
        message = self.get_next_raw_message(block)
        if message is not None:
            communication_state = json.loads(message)
            self.last_error = communication_state.get("error", None)
            self.game_is_ready = communication_state.get("ready_for_command")
            if self.last_error is None:
                self.in_game = communication_state.get("in_game")
                if self.in_game:
                    self.last_game_state = Game.from_json(communication_state.get("game_state"), communication_state.get("available_commands"))
            if perform_callbacks:
                if self.last_error is not None:
                    self.action_queue.clear()
                    if self.error_callback:
                        new_action = self.error_callback(self.last_error)
                        if new_action is not None:
                            self.add_action_to_queue(new_action)
                elif self.in_game:
                    if len(self.action_queue) == 0 and self.state_change_callback:
                        new_action = self.state_change_callback(self.last_game_state)
                        # 确保放入队列的是Action对象，而不是字符串
                        if new_action is not None:
                            self.add_action_to_queue(Action(new_action))
                elif self.stop_after_run:
                    self.clear_actions()
                elif self.out_of_game_callback:
                    new_action = self.out_of_game_callback()
                    # 确保放入队列的是Action对象，而不是字符串
                    if new_action is not None:
                        self.add_action_to_queue(Action(new_action))
            return True
        return False

    def run(self):
        """Start executing actions forever

        :return: None
        """
        while True:
            self.execute_next_action_if_ready()
            self.receive_game_state_update(perform_callbacks=True)

    def play_one_game(self, player_class: PlayerClass, ascension_level=0, seed=None):
        """

        :param player_class: the class to play
        :type player_class: PlayerClass
        :param ascension_level: the ascension level to use
        :type ascension_level: int
        :param seed: the alphanumeric seed to use
        :type seed: str
        :return: True if the game was a victory, else False
        :rtype: bool
        """
        self.clear_actions()
        while not self.game_is_ready:
            self.receive_game_state_update(block=True, perform_callbacks=False)
        if not self.in_game:
            StartGameAction(player_class, ascension_level, seed).execute(self)
            chinese_name = player_class.get_chinese_name()
            self.absolute_logger.write(f"开了一把新游戏，职业: {chinese_name}, 进阶等级: {ascension_level}, 种子: {seed}")
            self.receive_game_state_update(block=True)
        while self.in_game:
            self.execute_next_action_if_ready()
            self.receive_game_state_update()
        if self.last_game_state.screen_type == ScreenType.GAME_OVER:
            # 游戏结束界面，检查是否胜利，以及到了哪一层
            floor_reached = self.last_game_state.floor
            self.absolute_logger.write(f"游戏结束，最终达到层数: {floor_reached}")
            # 将高分写入高分文件（封装成方法）
            self._update_high_scores(player_class, ascension_level, floor_reached, self.last_game_state.screen.victory)
            return self.last_game_state.screen.victory
        else:
            return False

    # 新增：封装 high_scores.json 的读取、更新与写入逻辑
    def _update_high_scores(self, player_class: PlayerClass, ascension_level: int, floor_reached: int, victory: bool):
        """
        将本局结果写入 high_scores.json，保证以 utf-8 写入并使用 ensure_ascii=False 保留中文。
        不会抛异常（仅记录到 absolute_logger）。
        """
        
        scores_path = os.path.join(get_root_dir(), "high_scores.json")
        try:
            # 读取已有数据（容错）
            if os.path.exists(scores_path):
                with open(scores_path, 'r', encoding='utf-8') as f:
                    try:
                        high_scores = json.load(f)
                    except Exception:
                        high_scores = {}
            else:
                high_scores = {}

            class_name = player_class.get_chinese_name()
            ascension_str = "进阶 " + str(ascension_level)
            if class_name not in high_scores:
                high_scores[class_name] = {}
            if ascension_str not in high_scores[class_name]:
                high_scores[class_name][ascension_str] = {'最高抵达层数': 0, '连胜纪录': 0}

            if floor_reached > high_scores[class_name][ascension_str]['最高抵达层数']:
                high_scores[class_name][ascension_str]['最高抵达层数'] = floor_reached
                try:
                    self.absolute_logger.write(f"新的最高纪录！职业: {class_name}, 进阶等级: {ascension_level}, 最高层数: {floor_reached}")
                except Exception:
                    pass

            if victory:
                high_scores[class_name][ascension_str]['连胜纪录'] += 1
                try:
                    self.absolute_logger.write(f"当前连胜纪录: {high_scores[class_name][ascension_str]['连胜纪录']}")
                except Exception:
                    pass
            else:
                high_scores[class_name][ascension_str]['连胜纪录'] = 0

            # 写回文件（utf-8 + 保留中文）
            try:
                with open(scores_path, 'w', encoding='utf-8') as f:
                    json.dump(high_scores, f, ensure_ascii=False, indent=4)
            except Exception as e:
                try:
                    self.absolute_logger.write(f"写入 high_scores.json 失败: {e}")
                except Exception:
                    pass
        except Exception as e:
            try:
                self.absolute_logger.write(f"处理 high_scores.json 时发生异常: {e}")
            except Exception:
                pass
