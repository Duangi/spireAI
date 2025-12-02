import torch
import json
import sys
from typing import Optional
from spirecomm.ai import absolute_logger
from spirecomm.ai.dqn_core.algorithm import DQN
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.ai.dqn_core.reward import RewardCalculator
from spirecomm.spire import game
from spirecomm.spire.character import PlayerClass
from spirecomm.ai.progress_logger import ProgressLogger
from spirecomm.ai.absolute_logger import AbsoluteLogger
from spirecomm.spire.game import Game
from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases
from spirecomm.ai.dqn_core.action import DecomposedActionType


class DQNAgent:
    """
    一个集决策、记忆、学习于一体的DQN智能体。
    这个类是与spirecomm协调器直接交互的接口。
    """

    def __init__(self, play_mode=False, model_path=None):
        self.play_mode = play_mode
        # 1. 初始化核心组件
        self.state_processor = GameStateProcessor()
        self.reward_calculator = RewardCalculator()
        self.progress_logger = ProgressLogger()
        self.absolute_logger = AbsoluteLogger()
        self.absolute_logger.start_episode()

        # 假设状态向量大小为 10358
        state_size = 10358 
        self.dqn_algorithm = DQN(state_size, self.state_processor)

        if self.play_mode:
            self.dqn_algorithm.set_inference_mode()
            self.dqn_algorithm.policy_net.eval() # 游玩模式下，使用评估模式
        else:
            # 确保在训练模式下，网络处于 .train() 状态
            # 这将允许梯度计算和权重更新
            self.dqn_algorithm.policy_net.train()

        # 2. 用于存储上一步信息的变量
        self.previous_game_state = None
        self.previous_action = None
        self.previous_state_tensor = None

        # 调试标识：便于在日志/终端追踪此Agent实例的回调触发
        try:
            self.absolute_logger.write({"info": f"DQNAgent initialized (id={id(self)}, play_mode={self.play_mode})"})
        except Exception:
            # 如果 logger 尚未就绪，静默失败以免抛异常
            pass

    def get_next_action_in_game(self, game_state:Game):
        """
        这是由Coordinator在游戏状态改变时调用的核心回调函数。
        """
        # 调试输出：确认 in-game 回调被触发
        try:
            self.absolute_logger.write({
				"debug": f"get_next_action_in_game called (agent_id={id(self)}): floor={getattr(game_state,'floor',None)} "
						 f"in_game={getattr(game_state,'in_game',None)} in_combat={getattr(game_state,'in_combat',None)}"
			})
        except Exception:
            try:
                self.absolute_logger.write({"debug": "[DEBUG] get_next_action_in_game called (agent_id unknown)"})
            except Exception:
                pass

        # --- 学习与记忆 ---
        reward = 0
        # 只有在非游玩模式下，才进行学习
        if not self.play_mode:
            if self.previous_game_state is not None and self.previous_action is not None:
                # a. 计算奖励
                reward = self.reward_calculator.calculate(self.previous_game_state, game_state, self.previous_action)
                
                # b. 处理新状态
                next_state_tensor = self.state_processor.process(game_state)
                # game_state 是 Game 对象，直接访问属性
                done = not game_state.in_game
                
                # c. 记忆经验
                self.dqn_algorithm.remember(self.previous_state_tensor, self.previous_action, reward, next_state_tensor, done)
                
                # d. 训练模型
                self.dqn_algorithm.train()

        # --- 决策 ---
        # 1. 获取当前状态的向量和合法的动作掩码
        current_state_tensor = self.state_processor.process(game_state)
        # game_state 是 Game 对象，直接访问属性
        # 统一读取一次 available_commands 并在后续所有日志/判定中复用，确保一致性
        available_commands = game_state.available_commands
        masks = self.state_processor.get_action_masks(game_state)

        # 2. 如果没有可选动作，直接返回
        if not available_commands:
            # 游戏结束或出现意外情况，结束日志记录
            if self.progress_logger.file_handle:
                self.progress_logger.end_episode()
            return None

        # 3. 使用DQN算法选择一个动作
        try:
            chosen_action = self.dqn_algorithm.choose_action(current_state_tensor, masks)
        except RuntimeError as e:
            # 捕获如 "probability tensor contains either `inf`, `nan` or element < 0" 的异常，避免进程崩溃
            # 记录异常与上下文到日志，直接返回 None（不做回退）
            try:
                self.absolute_logger.write({
                    "error": f"dqn_algorithm.choose_action raised RuntimeError: {e}",
                    "context": {
                        "available_commands": list(available_commands or []),
                        "masks_repr": str(type(masks))
                    }
                })
            except Exception:
                pass
            return None
        # 保障：如果 choose_action 返回 None（未抛异常但未选到动作），也做回退处理
        if chosen_action is None:
            # 记录 choose_action 返回 None 的情况及上下文，直接返回 None（不做回退）
            try:
                self.absolute_logger.write({
                    "error": "dqn_algorithm.choose_action returned None",
                    "context": {
                        "available_commands": list(available_commands or []),
                        "masks_repr": str(type(masks))
                    }
                })
            except Exception:
                pass
            return None

        # ---------- 新增：记录决策时的 masks、可选命令与最终选中项，写入 absolute_logger 以便查看 ----------
        try:
            # 可用命令
            available = available_commands or []

            # masks 序列化（支持 dict / torch tensor / numpy / list / 其它）
            masks_serial = None
            masks_summary = None
            try:
                if isinstance(masks, dict):
                    masks_serial = {}
                    masks_summary = {}
                    for k, v in masks.items():
                        try:
                            # 将 numpy / torch / list 等转换为普通 Python 列表
                            if hasattr(v, "tolist"):
                                lst = v.tolist()
                            else:
                                lst = list(v)
                            # 确保列表为 bool/int 形式（例如 numpy.bool_ 仍可 bool()）
                            bool_list = [bool(x) for x in lst]
                            masks_serial[str(k)] = bool_list
                            allowed_idx = [i for i, flag in enumerate(bool_list) if flag]
                            masked_idx = [i for i, flag in enumerate(bool_list) if not flag]
                            masks_summary[str(k)] = {
                                "allowed_count": len(allowed_idx),
                                "masked_count": len(masked_idx),
                                "allowed_indices": allowed_idx,
                                "masked_indices": masked_idx
                            }
                        except Exception as e_inner:
                            masks_serial[str(k)] = f"serialize_error: {e_inner}"
                            masks_summary[str(k)] = {"error": str(e_inner)}
                else:
                    # 支持 torch/numpy/list 等
                    if hasattr(masks, "tolist"):
                        masks_serial = masks.tolist()
                    else:
                        try:
                            masks_serial = list(masks)
                        except Exception:
                            masks_serial = str(masks)
            except Exception as e:
                masks_serial = f"masks_serialize_error: {e}"

            # 把 masks_summary 的索引转换为可读含义（只保留 summary，去掉过长的 raw 列表）
            masks_summary_human = None
            try:
                if isinstance(masks_summary, dict):
                    masks_summary_human = {}
                    for key, info in masks_summary.items():
                        if not isinstance(info, dict) or "allowed_indices" not in info:
                            masks_summary_human[key] = info
                            continue
                        allowed = info.get("allowed_indices", [])
                        # key 对应不同含义的索引映射
                        if key == "action_type":
                            mapped = []
                            for idx in allowed:
                                try:
                                    mapped.append(DecomposedActionType(idx).name.lower())
                                except Exception:
                                    mapped.append(str(idx))
                            masks_summary_human[key] = {"allowed": mapped, "allowed_count": len(mapped)}
                        elif key == "play_card":
                            masks_summary_human[key] = {"allowed": [f"hand_{i}" for i in allowed], "allowed_count": len(allowed)}
                        elif key == "target_monster":
                            masks_summary_human[key] = {"allowed": [f"monster_{i}" for i in allowed], "allowed_count": len(allowed)}
                        elif key == "choose_option":
                            masks_summary_human[key] = {"allowed": [f"choice_{i}" for i in allowed], "allowed_count": len(allowed)}
                        elif key == "potion":
                            masks_summary_human[key] = {"allowed": [f"potion_{i}" for i in allowed], "allowed_count": len(allowed)}
                        else:
                            masks_summary_human[key] = {"allowed_indices": allowed, "allowed_count": len(allowed)}
                else:
                    masks_summary_human = masks_summary
            except Exception:
                masks_summary_human = masks_summary

            # 尝试获取所有合法动作的 Q 值（可能开销较大，但对排查非常有用）
            try:
                all_q = self.dqn_algorithm.get_all_legal_action_q_values(current_state_tensor, game_state)
            except Exception as e:
                all_q = f"get_all_legal_action_q_values error: {e}"

            # 提取关键词集合（动作的第一个 token），便于与 available_commands 比对
            legal_actions = set()
            legal_keywords = set()
            if isinstance(all_q, dict):
                legal_actions = set(all_q.keys())
                legal_keywords = {str(a).split()[0].lower() for a in legal_actions if isinstance(a, str) and a}
            # 为日志过滤掉一些界面控制类命令（这些不会作为可执行动作发送）
            _filter_out = {"key", "click", "wait", "state"}
            available_for_log = [a for a in available if str(a).lower() not in _filter_out]
            # 用于日志的 keywords（过滤后）
            available_keywords_log = sorted(list({str(a).split()[0].lower() for a in available_for_log}))
            # 仍保留一个用于比较的完整 keywords 集（不受过滤影响）
            available_keywords = {str(a).split()[0].lower() for a in available}
 
            # 不一致项
            legal_but_not_available = sorted(list(legal_keywords - available_keywords))
            available_but_not_legal = sorted(list(available_keywords - legal_keywords))
 
            chosen_str = chosen_action.to_string() if hasattr(chosen_action, "to_string") else str(chosen_action)
            chosen_keyword = chosen_str.split()[0].lower() if isinstance(chosen_str, str) and chosen_str else ""
 
            decision_debug = {
                # 日志中仅输出过滤后的 available_commands/keywords，去掉界面控制项
                "available_commands": available_for_log,
                "available_keywords": available_keywords_log,
                "masks_summary": masks_summary_human,
                "chosen_action": chosen_str,
                "chosen_keyword": chosen_keyword,
                "q_values": all_q,
                "legal_actions": sorted(list(legal_actions)) if isinstance(all_q, dict) else all_q,
                "legal_keywords": sorted(list(legal_keywords)),
                "legal_but_not_available": legal_but_not_available,
                "available_but_not_legal": available_but_not_legal,
                "chosen_allowed_by_available": (chosen_keyword in available_keywords),
                "chosen_present_in_legal": (chosen_str in legal_actions) if isinstance(all_q, dict) else None
            }
            # 写入 absolute logger（会输出到文件）
            try:
                self.absolute_logger.write({"decision_debug": decision_debug})
            except Exception:
                pass
        except Exception as e:
            try:
                self.absolute_logger.write({"debug": f"记录 decision_debug 时异常: {e}"})
            except Exception:
                pass
        # ------------------------------------------------------------------------------------

        # --- 可视化日志记录 ---
        if self.previous_game_state is not None:
            # 获取所有合法动作的Q值用于记录
            q_values_log = self.dqn_algorithm.get_all_legal_action_q_values(current_state_tensor, game_state)

            log_info = {
                'q_values': q_values_log,
                'action_taken_at_prev_state': self.previous_action.to_string() if self.previous_action else "None",
                'reward_for_prev_action': reward,
                'chosen_action_for_current_state': chosen_action.to_string(),
                'prev_player': self.previous_game_state.player.to_json() if self.previous_game_state.player else {},
                'next_player': game_state.player.to_json() if game_state.player else {},
                'prev_monsters': [m.to_json() for m in self.previous_game_state.monsters],
                'next_monsters': [m.to_json() for m in game_state.monsters],
                'reward': reward
            }
            self.progress_logger.log_step(log_info)
            self.absolute_logger.write(log_info)
        # --- 为下一步做准备 ---
        # 存储当前的状态和动作用于下一次学习
        self.previous_game_state = game_state
        self.previous_action = chosen_action
        self.previous_state_tensor = current_state_tensor

        # --- 返回动作给协调器 ---
        # 核心改动：协调器需要的是一个可执行的字符串命令，而不是我们的内部动作对象。
        # 我们调用 to_string() 方法将其转换。
        action_string = chosen_action.to_string()
        # 新增：校验 action_string 的动作类型是否在当前可用命令里（避免发送游戏当前不接受的命令）
        try:
            available = getattr(game_state, "available_commands", []) or []
            # 取第一个 token 作为命令关键词（例如 "play 4 0" -> "play"）
            action_keyword = action_string.split()[0].lower() if isinstance(action_string, str) and action_string else ""
            available_lower = [str(a).lower() for a in available]
            if action_keyword not in available_lower:
                # 记录警告并尝试回退到安全命令
                try:
                    self.absolute_logger.write({
                        "warn": f"Attempt to send invalid action '{action_string}' with keyword '{action_keyword}', available={available}"
                    })
                except Exception:
                    pass

                # 回退策略：优先使用 'proceed'，否则返回 None（不发送命令）
                if "proceed" in available_lower:
                    action_string = "proceed"
                elif available:
                    # 若没有 'proceed'，尝试简单返回第一个可用命令（不带参数）
                    fallback = available[0]
                    action_string = str(fallback)
                else:
                    # 无可用命令，放弃发送
                    return None
        except Exception as e:
            try:
                self.absolute_logger.write({"debug": f"校验动作可用性时异常: {e}"})
            except Exception:
                pass

        try:
            self.absolute_logger.write({"debug": f"get_next_action_in_game returning action_string: {action_string}"})
        except Exception:
            pass
        
        return action_string

    def get_next_action_out_of_game(self, game_state):
        # 在游戏外，我们总是选择开始游戏
        # 调试输出：确认 out-of-game 回调被触发
        try:
            self.absolute_logger.write({
				"debug": f"get_next_action_out_of_game called (agent_id={id(self)}): floor={getattr(game_state,'floor',None)} "
						 f"in_game={getattr(game_state,'in_game',None)} in_combat={getattr(game_state,'in_combat',None)}"
			})
        except Exception:
            try:
                self.absolute_logger.write({"debug": "[DEBUG] get_next_action_out_of_game called (agent_id unknown)"})
            except Exception:
                pass

        # 如果上一个日志文件还开着，说明游戏异常结束，关闭它
        if self.progress_logger.file_handle:
            self.progress_logger.end_episode()
        self.previous_game_state = None
        self.previous_action = None
        self.previous_state_tensor = None
        self.progress_logger.start_episode() # 新的一局游戏开始
        self.absolute_logger.start_episode()
        self.absolute_logger.write({'--------------action': 'start_game---------------------'})
        start_action_obj = self.state_processor.get_start_game_action(game_state)

        try:
            if hasattr(start_action_obj, "to_string"):
                action_string = start_action_obj.to_string()
            else:
                action_string = str(start_action_obj)
        except Exception:
            action_string = str(start_action_obj)

        # 同样校验 out-of-game 的动作是否在 available_commands 中（有些状态 coordinator/game 也会返回限制）
        try:
            available = getattr(game_state, "available_commands", []) or []
            action_keyword = action_string.split()[0].lower() if isinstance(action_string, str) and action_string else ""
            if available and action_keyword not in [str(a).lower() for a in available]:
                try:
                    self.absolute_logger.write({
                        "warn": f"Out-of-game invalid action '{action_string}', available={available}"
                    })
                except Exception:
                    pass
                if "proceed" in [str(a).lower() for a in available]:
                    action_string = "proceed"
                else:
                    return None
        except Exception as e:
            try:
                self.absolute_logger.write({"debug": f"Out-of-game 校验动作可用性时异常: {e}"})
            except Exception:
                pass

        try:
            self.absolute_logger.write({"debug": f"get_next_action_out_of_game returning action_string: {action_string}"})
        except Exception:
            pass

        return action_string

    def handle_error(self, error):
        """
        处理来自协调器的错误回调。
        当一个动作无效时，这通常意味着AI对游戏状态的理解出现了偏差。
        一个稳健的策略是清空动作队列，并根据当前最新的游戏状态重新决策。
        """
        self.absolute_logger.write({"error": f"Received error: {error}"})
        # 返回 None 会导致协调器清空动作队列，然后在下一个循环中根据最新状态重新调用 get_next_action_in_game
        return None

    def change_class(self, chosen_class: PlayerClass):
        # 这个方法可以被主循环调用，但目前我们的Agent是通用的，所以不需要做什么
        self.absolute_logger.write({"changing_class": f"Changing class to {chosen_class.name}"})

    def learn(self, batch_size=32):
        """
        从经验回放区采样数据，训练一次网络。
        这是提供给 train.py 在每局结束后调用的接口。
        """
        self.dqn_algorithm.train(batch_size)