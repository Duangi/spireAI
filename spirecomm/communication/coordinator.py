import sys
import queue
import threading
import json
import collections
import os



from spirecomm.spire.game import Game
from spirecomm.spire.screen import ScreenType
from spirecomm.communication.action import Action, StartGameAction
from spirecomm.spire.character import PlayerClass
from spirecomm.utils.path import get_root_dir
from filelock import FileLock

def read_stdin(input_queue, coordinator):
    """Read lines from stdin and write them to a queue

    :param input_queue: A queue, to which lines from stdin will be written
    :type input_queue: queue.Queue
    :param coordinator: The coordinator instance
    :type coordinator: Coordinator
    :return: None
    """
    while True:
        stdin_input = ""
        while True:
            input_char = sys.stdin.read(1)
            if not input_char: # EOF
                # 父进程关闭了输入流，说明游戏可能已经退出
                if coordinator.on_exit_callback:
                    try:
                        coordinator.on_exit_callback()
                    except Exception as e:
                        # 忽略异常，继续退出
                        pass
                # 给 WandB 一点时间完成清理（最多等待 2 秒）
                try:
                    import time
                    time.sleep(2)
                except Exception:
                    pass
                # 直接强制终止当前 Python 进程
                os._exit(0)
            if input_char == '\n':
                break
            else:
                stdin_input += input_char
        # 输入到txt文件
        # with open('process.txt', 'a') as f:
        #     f.write('-----input-----' + '\n')
        #     f.write(stdin_input + '\n')
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
        # with open('process.txt', 'a') as f:
        #     f.write('-----output-----' + '\n')
        #     f.write(output + '\n')
        # print(output, end='\n', flush=True)
        sys.__stdout__.write(output + '\n')
        sys.__stdout__.flush()


class Coordinator:
    """An object to coordinate communication with Slay the Spire"""

    def __init__(self) -> None:
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.on_exit_callback = None # Initialize before starting threads
        self.input_thread = threading.Thread(target=read_stdin, args=(self.input_queue, self))
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

    def register_on_exit_callback(self, new_callback):
        """Register a function to be called when the game process exits (EOF on stdin)

        :param new_callback: the function to call
        :type new_callback: function() -> None
        :return: None
        """
        self.on_exit_callback = new_callback

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
                # 即使不在游戏中，如果有 game_state 数据，也尝试解析，以便获取 Game Over 等信息
                if "game_state" in communication_state and communication_state["game_state"]:
                    self.last_game_state = Game.from_json(communication_state.get("game_state"), communication_state.get("available_commands"))
                elif self.in_game: # Fallback: if in_game is true but no game_state (unlikely), or just to be safe
                     pass 

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
                    # Pass the last known game state (which might be the Game Over screen)
                    new_action = self.out_of_game_callback(self.last_game_state)
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
            self.receive_game_state_update(block=True)
        while self.in_game:
            self.execute_next_action_if_ready()
            self.receive_game_state_update()
        
        if self.last_game_state and self.last_game_state.screen_type == ScreenType.GAME_OVER:
            # 游戏结束界面，检查是否胜利，以及到了哪一层
            floor_reached = self.last_game_state.floor
            # 将高分写入高分文件（封装成方法）
            self._update_high_scores(player_class, ascension_level, floor_reached, self.last_game_state.screen.victory)
            return self.last_game_state.screen.victory
        else:
            return False

    # 新增：封装 high_scores.json 的读取、更新与写入逻辑
    def _update_high_scores(self, player_class: PlayerClass, ascension_level: int, floor_reached: int, victory: bool):
        """
        将本局结果写入 high_scores.json，保证以 utf-8 写入并使用 ensure_ascii=False 保留中文。
        不会抛异常
        """
        
        scores_path = os.path.join(get_root_dir(), "high_scores.json")
        lock_path = scores_path + ".lock"

        lock = FileLock(lock_path, timeout=10)
        try:
            with lock:
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

                if victory:
                    high_scores[class_name][ascension_str]['连胜纪录'] += 1
                else:
                    high_scores[class_name][ascension_str]['连胜纪录'] = 0

                # 写回文件（utf-8 + 保留中文）
                try:
                    with open(scores_path, 'w', encoding='utf-8') as f:
                        json.dump(high_scores, f, ensure_ascii=False, indent=4)
                except Exception as e:
                    pass
        except Exception as e:
            pass
