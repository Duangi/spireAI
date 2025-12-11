from spirecomm.ai.dqn_core.algorithm import SpireAgent
from spirecomm.ai.dqn_core.model import SpireConfig
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.ai.dqn_core.reward import RewardCalculator
from spirecomm.communication.action import StartGameAction
from spirecomm.spire import game
from spirecomm.spire.character import PlayerClass
from spirecomm.spire.game import Game
from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases
from spirecomm.ai.dqn_core.action import DecomposedActionType
from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
import torch


class DQNAgent:
    """
    一个集决策、记忆、学习于一体的DQN智能体。
    这个类是与spirecomm协调器直接交互的接口。
    """

    def __init__(self, play_mode=False, model_path=None, wandb_logger: WandbLogger = None, memory_callback=None):
        self.play_mode = play_mode
        self.memory_callback = memory_callback
        # 1. 初始化核心组件
        self.state_processor = GameStateProcessor()
        self.reward_calculator = RewardCalculator(state_processor=self.state_processor)

        # 初始化配置
        config = SpireConfig()
        # 初始化 SpireAgent
        self.dqn_algorithm = SpireAgent(config, wandb_logger=wandb_logger)

        if self.play_mode:
            self.dqn_algorithm.set_inference_mode()
            self.dqn_algorithm.policy_net.eval() # 游玩模式下，使用评估模式
        else:
            # 确保在训练模式下，网络处于 .train() 状态
            # 这将允许梯度计算和权重更新
            self.dqn_algorithm.policy_net.train()

        # 2. 用于存储上一步信息的变量
        self.previous_game_state = None
        self.previous_prev_state = None # 上上一步状态，用于检测卡bug
        self.previous_action = None
        self.previous_state_tensor = None

        # 如果提供了预训练模型路径，尝试加载
        if model_path:
            try:
                self.load_model(model_path)
            except Exception as e:
                raise RuntimeError(f"无法加载模型: {e}")
    def get_next_action_in_game(self, game_state:Game):
        """
        这是由Coordinator在游戏状态改变时调用的核心回调函数。
        """
        # --- 学习与记忆 ---
        reward = 0
        # 只有在非游玩模式下，或者设置了 memory_callback 时，才进行记忆
        if not self.play_mode or self.memory_callback:
            if self.previous_game_state is not None and self.previous_action is not None:
                # a. 计算奖励
                reward, reward_details = self.reward_calculator.calculate(self.previous_game_state, game_state, self.previous_action, self.previous_prev_state)
                
                # b. 处理新状态
                next_state_tensor = self.state_processor.get_state_tensor(game_state)
                # game_state 是 Game 对象，直接访问属性
                # 在 get_next_action_in_game 中，我们肯定还在游戏中，所以 done 强制为 False
                # 真正的 done=True 只会在 get_next_action_out_of_game 中触发
                done = False
                
                # c. 记忆经验
                if self.memory_callback:
                    self.memory_callback(self.previous_state_tensor, self.previous_action, reward, next_state_tensor, done, reward_details, self.previous_game_state, game_state, self.previous_prev_state)
                elif not self.play_mode:
                    self.dqn_algorithm.remember(self.previous_state_tensor, self.previous_action, reward, next_state_tensor, done, reward_details)
                    # d. 训练模型
                    self.dqn_algorithm.train()

        # --- 决策 ---
        # 1. 获取当前状态的向量和合法的动作掩码
        current_state_tensor = self.state_processor.get_state_tensor(game_state)
        # game_state 是 Game 对象，直接访问属性
        # 统一读取一次 available_commands 并在后续所有日志/判定中复用，确保一致性
        available_commands = game_state.available_commands
        masks = self.state_processor.get_action_masks(game_state)

        # 3. 使用DQN算法选择一个动作
        # SpireAgent.choose_action 需要 masks 参数
        chosen_action = self.dqn_algorithm.choose_action(current_state_tensor, masks, game_state)
        

        # --- 为下一步做准备 ---
        # 存储当前的状态和动作用于下一次学习
        self.previous_prev_state = self.previous_game_state
        self.previous_game_state = game_state

        self.previous_action = chosen_action
        self.previous_state_tensor = current_state_tensor

        # --- 返回动作给协调器 ---
        # 核心改动：协调器需要的是一个可执行的字符串命令，而不是我们的内部动作对象。
        # 我们调用 to_string() 方法将其转换。
        action_string = chosen_action.to_string()
        
        return action_string

    def get_next_action_out_of_game(self, final_game_state=None):
        # 处理上一局游戏的最后一步
        if (not self.play_mode or self.memory_callback) and self.previous_game_state is not None and self.previous_action is not None:
            # 计算最后一步的奖励
            # 使用 final_game_state 作为 next_state (如果可用)
            # 如果 final_game_state 是 Game Over 屏幕，它应该包含最终信息
            reward, reward_details = self.reward_calculator.calculate(self.previous_game_state, final_game_state, self.previous_action, self.previous_prev_state)
            
            # 记录经验，标记 done=True
            # next_state_tensor: 如果有 final_game_state，尝试转换它，否则用全0
            if final_game_state:
                try:
                    next_state_tensor = self.state_processor.get_state_tensor(final_game_state)
                except:
                    next_state_tensor = torch.zeros_like(self.previous_state_tensor)
            else:
                next_state_tensor = torch.zeros_like(self.previous_state_tensor)
            
            if self.memory_callback:
                self.memory_callback(self.previous_state_tensor, self.previous_action, reward, next_state_tensor, True, reward_details, self.previous_game_state, final_game_state, self.previous_prev_state)
            elif not self.play_mode:
                self.dqn_algorithm.remember(self.previous_state_tensor, self.previous_action, reward, next_state_tensor, True, reward_details)
                # 触发一次训练
                self.dqn_algorithm.train()
            
        # 重置状态
        self.previous_game_state = None
        self.previous_prev_state = None
        self.previous_action = None
        self.previous_state_tensor = None

        return StartGameAction(self.chosen_class)

    def handle_error(self, error):
        pass

    def change_class(self, new_class:PlayerClass):
        self.chosen_class = new_class

    def learn(self, batch_size=32):
        """
        从经验回放区采样数据，训练一次网络。
        这是提供给 train.py 在每局结束后调用的接口。
        """
        # SpireAgent.train() 使用内部配置的 batch_size
        self.dqn_algorithm.train()

    def load_model(self, model_path: str, map_location=None):
        """
        加载 model_path 中的 state_dict 到 policy_net（并尝试同步到 target_net）。
        不会改变 play_mode 的值
        """
        state_dict = torch.load(model_path, map_location=map_location)
        self.dqn_algorithm.policy_net.load_state_dict(state_dict)
        # 尝试同步到 target_net
        self.dqn_algorithm.update_target_net()
