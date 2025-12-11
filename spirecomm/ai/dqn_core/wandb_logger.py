import wandb
import torch
import numpy as np
import os
import subprocess
import socket
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union
from spirecomm.ai.dqn_core.model import SpireState
from spirecomm.utils.data_processing import ID_TO_TEXT
from spirecomm.ai.dqn_core.action import ActionType
import signal
import atexit
import threading
import time

class SpireStateDecoder:
    """
    负责将 SpireState Tensor 解码为人类可读的字典结构。
    依赖于 spirecomm.utils.data_processing.ID_TO_TEXT 的实时更新。
    """
    
    @staticmethod
    def decode_id(idx: Union[int, torch.Tensor]) -> str:
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        idx = int(idx)
        if idx == 0:
            return ""
        return ID_TO_TEXT.get(idx, f"UNK_{idx}")

    @staticmethod
    def decode_ids(ids: torch.Tensor) -> List[str]:
        # ids: [N] or [1, N]
        if ids.dim() > 1:
            ids = ids.squeeze(0)
        return [SpireStateDecoder.decode_id(i) for i in ids.flatten().tolist() if i != 0]

    @staticmethod
    def decode(state: SpireState, batch_idx: int = 0) -> Dict[str, Any]:
        """
        解码单个样本的状态。
        """
        # Helper to get tensor data for a specific batch index
        def get(tensor):
            if tensor is None: return None
            if tensor.dim() > 1 and tensor.shape[0] > batch_idx:
                return tensor[batch_idx]
            return tensor

        s = {}
        
        # --- 1. Global Numeric ---
        # Layout: MaxHP(1), CurHP(1), Ratio(1), Floor(1), Act(4), Gold(1), Class(4), Ascension(1), Boss(3)
        g_num = get(state.global_numeric).cpu().tolist()
        
        # Act (indices 4-7)
        act_idx = np.argmax(g_num[4:8]) + 1
        # Class (indices 9-12)
        class_idx = np.argmax(g_num[9:13])
        class_names = ["Ironclad", "Silent", "Defect", "Watcher"]
        class_name = class_names[class_idx] if 0 <= class_idx < 4 else "Unknown"
        
        s['Global'] = {
            'HP_Norm': f"{g_num[1]:.2f}/{g_num[0]:.2f}", # Normalized values
            'Floor_Norm': f"{g_num[3]:.2f}",
            'Act': act_idx,
            'Class': class_name,
            'Gold_Log': f"{g_num[8]:.2f}"
        }

        # --- 2. Cards ---
        s['Hand'] = SpireStateDecoder.decode_ids(get(state.hand_ids))
        s['Deck_Count'] = (get(state.deck_ids) != 0).sum().item()
        s['Draw_Count'] = (get(state.draw_pile_ids) != 0).sum().item()
        s['Discard_Count'] = (get(state.discard_pile_ids) != 0).sum().item()
        
        # --- 3. Monsters ---
        monsters = []
        m_ids = get(state.monster_ids).cpu().tolist()
        m_int_ids = get(state.monster_intent_ids).cpu().tolist()
        
        for i, m_id in enumerate(m_ids):
            if m_id == 0: continue
            name = SpireStateDecoder.decode_id(m_id)
            intent = SpireStateDecoder.decode_id(m_int_ids[i])
            monsters.append(f"{name} [{intent}]")
        s['Monsters'] = monsters

        # --- 4. Relics & Potions ---
        s['Relics'] = SpireStateDecoder.decode_ids(get(state.relic_ids))
        s['Potions'] = SpireStateDecoder.decode_ids(get(state.potion_ids))
        
        # --- 5. Screen ---
        s['ScreenItems'] = SpireStateDecoder.decode_ids(get(state.screen_item_ids))
        
        # --- 6. Player Powers ---
        p_powers = []
        p_pwr_ids = get(state.player_power_ids).cpu().tolist()
        for pid in p_pwr_ids:
            if pid == 0: continue
            p_powers.append(SpireStateDecoder.decode_id(pid))
        s['PlayerPowers'] = p_powers

        return s

class WandbLogger:
    def __init__(self, project_name="spire-ai", run_name=None, config=None):
        """
        初始化 Wandb Logger。
        :param project_name: Wandb 项目名称
        :param run_name: 本次运行的名称
        :param config: 超参数字典
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.api_key = os.getenv("WANDB_API_KEY")
        self.enabled = False

        if self.api_key and self.api_key.strip():
            # Pre-check connectivity to avoid hanging in retry loop
            if not self._check_connection():
                os.environ["WANDB_MODE"] = "offline"

            try:
                # Try online login first
                try:
                    # If WANDB_MODE is already offline, login might be skipped or behave differently
                    if os.environ.get("WANDB_MODE") != "offline":
                        wandb.login(key=self.api_key)
                    
                    self.run = wandb.init(project=project_name, name=run_name, config=config, reinit=True)
                    self.enabled = True
                except Exception as e:
                    # Fallback to offline mode
                    os.environ["WANDB_MODE"] = "offline"
                    self.run = wandb.init(project=project_name, name=run_name, config=config, reinit=True)
                    self.enabled = True
            except Exception as e:
                self.enabled = False
        else:
            self.enabled = False

        self.step_buffer = []
        # 定义 Rich Table 的列
        self.columns = [
            "Step", 
            "Action", 
            "Reward",
            "Reward Details",
            "Type Q",
            "Card Q",
            "Monster Q",
            "Choice Q",
            "PotUse Q",
            "PotDisc Q",
            "HP",
            "Gold",
            "Floor",
            "Act",
            "Hand", 
            "Monsters", 
            "Player Powers",
            "Class"
        ]
        # 退出/终止管理
        self._terminate_lock = threading.Lock()
        self._terminated = False
        self._register_exit_handlers()

    def _check_connection(self, host="api.wandb.ai", port=443, timeout=3):
        """
        Simple socket check to see if we can reach WandB API.
        """
        # If API key is present, assume we want to try connecting.
        # The socket check is unreliable with proxies.
        if self.api_key:
            return True
            
        try:
            # If proxy is set, we can't easily check via socket without implementing proxy protocol.
            # But usually if proxy is set correctly, socket connect to proxy might work, but here we try direct.
            # If direct fails, it might be blocked.
            # Actually, a better check is to use requests if available, or just rely on the fact that
            # if this fails, we go offline.
            # However, if user uses a proxy, direct socket to api.wandb.ai might fail even if proxy works.
            # So we should be careful.
            # Let's try to use requests if possible, as it respects env vars.
            import requests
            try:
                requests.get(f"https://{host}", timeout=timeout)
                return True
            except:
                return False
        except ImportError:
            # Fallback to socket if requests not available (unlikely in this env)
            try:
                socket.create_connection((host, port), timeout=timeout)
                return True
            except OSError:
                return False

    def _register_exit_handlers(self):
        """
        注册 signal 和 atexit 回调，确保程序异常退出或被终止时尝试 flush/finish/sync wandb。
        这些回调必须尽量简短且容错。
        """
        try:
            atexit.register(self._on_terminate)
        except Exception:
            pass

        try:
            # 在多数平台上监听 SIGINT/SIGTERM
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(sig, lambda signum, frame: self._on_terminate())
                except Exception:
                    # 某些平台（如 Windows）对部分信号的支持有限，忽略错误
                    pass
        except Exception:
            pass

    def _on_terminate(self):
        """
        进程收到终止信号或正常退出时调用：尝试 finish run。
        注意：不在这里提交表格，因为异常退出时数据可能不完整。
        所有步骤均捕获异常以避免抛出。
        """
        # 保证只执行一次
        try:
            with self._terminate_lock:
                if getattr(self, "_terminated", False):
                    return
                self._terminated = True
        except Exception:
            # 如果锁不可用也继续尝试执行一次性清理
            try:
                if getattr(self, "_terminated", False):
                    return
                self._terminated = True
            except Exception:
                pass

        try:
            # 尝试正常结束 run（不提交缓冲区中的数据，避免产生不完整的表格）
            if getattr(self, "enabled", False):
                try:
                    self.finish()
                except Exception:
                    pass
        except Exception:
            pass

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        实时记录标量指标 (Loss, Reward, Epsilon 等)
        """
        if not self.enabled:
            return
        wandb.log(metrics, step=step)

    def log_step(self, 
                 step_count: int,
                 state: SpireState, 
                 action_desc: str, 
                 q_values: Optional[Dict[str, Dict[str, float]]], 
                 reward: float,
                 reward_details: str = ""):
        """
        记录一步决策的详细信息到缓冲区，用于生成 Rich Table。
        """
        if not self.enabled:
            return
            
        decoded = SpireStateDecoder.decode(state)
        
        # Helper to format top 3 Q-values
        def format_top3(q_dict):
            if not q_dict: return "-"
            # Filter out masked values
            valid_q = {k: v for k, v in q_dict.items() if v > -1e8}
            if not valid_q: return "-"
            # Sort by value descending
            top_q = sorted(valid_q.items(), key=lambda item: item[1], reverse=True)[:3]
            return ", ".join([f"{k}: {v:.2f}" for k, v in top_q])

        # Extract Q-values by category
        q_type_str = "-"
        q_card_str = "-"
        q_monster_str = "-"
        q_choice_str = "-"
        q_pot_use_str = "-"
        q_pot_disc_str = "-"

        if q_values:
            q_type_str = format_top3(q_values.get('action_type', {}))
            q_card_str = format_top3(q_values.get('play_card', {}))
            q_monster_str = format_top3(q_values.get('target_monster', {}))
            q_choice_str = format_top3(q_values.get('choose_option', {}))
            q_pot_use_str = format_top3(q_values.get('potion_use', {}))
            q_pot_disc_str = format_top3(q_values.get('potion_discard', {}))
        
        # Format State Info
        # decoded['Global'] has: 'HP_Norm', 'Floor_Norm', 'Act', 'Class', 'Gold_Log'
        # We try to make them more readable if possible, but currently we use what we have.
        # Note: Floor is normalized by /50, Gold is log(gold+1).
        # To make it readable, we can try to reverse it roughly for display.
        
        try:
            floor_val = float(decoded['Global']['Floor_Norm']) * 50
            floor_str = f"{int(round(floor_val))}"
        except:
            floor_str = decoded['Global']['Floor_Norm']

        try:
            # Gold is normalized by linear clip to 1000
            gold_val = float(decoded['Global']['Gold_Log']) * 1000
            gold_str = f"{int(round(gold_val))}"
        except:
            gold_str = decoded['Global']['Gold_Log']

        act_str = str(decoded['Global']['Act'])
        class_str = str(decoded['Global']['Class'])
        hp_str = str(decoded['Global']['HP_Norm'])

        hand_str = ", ".join(decoded['Hand'])
        monster_str = ", ".join(decoded['Monsters']) # Changed from <br> to comma for cleaner text view
        power_str = ", ".join(decoded['PlayerPowers'])
        
        row = [
            step_count,
            action_desc,
            reward,
            reward_details,
            q_type_str,
            q_card_str,
            q_monster_str,
            q_choice_str,
            q_pot_use_str,
            q_pot_disc_str,
            hp_str,
            gold_str,
            floor_str,
            act_str,
            hand_str,
            monster_str,
            power_str,
            class_str
        ]
        self.step_buffer.append(row)

    def commit_table(self, table_name="Episode_Trace", clear_buffer=True, step=None):
        """
        将缓冲区的内容提交为 Wandb Table。通常在一个 Episode 结束时调用。
        :param clear_buffer: 是否在提交后清空缓冲区。
                             如果为 False，则可以实现"实时更新"的效果（每次提交包含之前所有行）。
        :param step: 当前步数，用于同步 WandB 的 step。
        """
        # 如果未启用或者没有数据，静默返回（不抛异常），避免流程中断
        if not self.enabled:
            return

        if not self.step_buffer:
            return

        try:
            table = wandb.Table(columns=self.columns, data=self.step_buffer)

            # 优先使用 run.log（更可靠地绑定到当前 run），回退到 wandb.log
            try:
                if hasattr(self, "run") and self.run is not None:
                    if step is not None:
                        self.run.log({table_name: table}, step=step)
                    else:
                        self.run.log({table_name: table})
                else:
                    if step is not None:
                        wandb.log({table_name: table}, step=step)
                    else:
                        wandb.log({table_name: table})
            except Exception:
                # 若 run.log 出错，退回到全局 wandb.log（再捕获一次以防万一）
                try:
                    if step is not None:
                        wandb.log({table_name: table}, step=step)
                    else:
                        wandb.log({table_name: table})
                except Exception:
                    # 最后降级为静默忽略，避免影响训练流程
                    pass

        except Exception:
            # 构建表或日志过程出错，不抛出，让调用者继续
            pass
        finally:
            if clear_buffer:
                self.step_buffer = []

    def finish(self):
        if not self.enabled:
            return

        # 如果还有未提交的表格，先尝试提交（不清空），确保离线文件中包含最新表格
        try:
            if self.step_buffer:
                try:
                    # 不清空 buffer，交由调用者控制
                    self.commit_table(clear_buffer=False)
                except Exception:
                    pass
        except Exception:
            pass

        # 1. Try normal finish with extended timeout
        try:
            if hasattr(self, "run") and self.run is not None:
                # 给 WandB 足够的时间完成清理
                self.run.finish(quiet=True)
        except Exception:
            pass
        
        # 2. 等待 WandB 后台进程完成
        try:
            time.sleep(1.5)
        except Exception:
            pass
        
        # 3. 如果是离线模式，尝试用 wandb CLI 同步正确的 run 目录到云端
        try:
            is_offline = (os.environ.get("WANDB_MODE") == "offline")
            try:
                if hasattr(self, "run") and self.run is not None:
                    settings_mode = getattr(getattr(self.run, "settings", None), "mode", None)
                    if settings_mode == "offline":
                        is_offline = True
            except Exception:
                pass

            if is_offline and hasattr(self, "run") and getattr(self.run, "dir", None):
                # 使用 self.run.dir（直接指向该 run 的目录），不要取 dirname，否则会错过 files
                run_dir = self.run.dir
                if os.path.exists(run_dir):
                    try:
                        subprocess.run(["wandb", "sync", run_dir], check=False, timeout=10)
                    except Exception:
                        pass
        except Exception:
            pass
