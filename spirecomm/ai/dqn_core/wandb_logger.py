import wandb
import torch
import numpy as np
import os
import subprocess
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Union
from spirecomm.ai.dqn_core.model import SpireState
from spirecomm.utils.data_processing import ID_TO_TEXT
from spirecomm.ai.dqn_core.action import ActionType

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
            try:
                # Try online login first
                try:
                    wandb.login(key=self.api_key)
                    self.run = wandb.init(project=project_name, name=run_name, config=config, reinit=True)
                    self.enabled = True
                    print(f"[WandbLogger] Successfully initialized run: {run_name}")
                except Exception as e:
                    print(f"[WandbLogger] Online initialization failed: {e}. Trying offline mode...")
                    # Fallback to offline mode
                    os.environ["WANDB_MODE"] = "offline"
                    self.run = wandb.init(project=project_name, name=run_name, config=config, reinit=True)
                    self.enabled = True
                    print(f"[WandbLogger] Successfully initialized run in OFFLINE mode: {run_name}")
            except Exception as e:
                print(f"[WandbLogger] Initialization failed: {e}")
                self.enabled = False
        else:
            print("[WandbLogger] WANDB_API_KEY not found or empty. Wandb logging disabled.")
            self.enabled = False

        self.step_buffer = []
        # 定义 Rich Table 的列
        self.columns = [
            "Step", 
            "Action", 
            "Reward",
            "Reward Details",
            "Top Q1",
            "Top Q2",
            "Top Q3",
            "HP",
            "Gold",
            "Floor",
            "Act",
            "Hand", 
            "Monsters", 
            "Player Powers",
            "Class"
        ]

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
                 q_values: Optional[Dict[str, float]], 
                 reward: float,
                 reward_details: str = ""):
        """
        记录一步决策的详细信息到缓冲区，用于生成 Rich Table。
        """
        if not self.enabled:
            return
            
        decoded = SpireStateDecoder.decode(state)
        
        # Format Q-values (Split into 3 columns)
        # Filter out masked values (e.g. -1e9 or -inf)
        valid_q = {}
        if q_values:
            for k, v in q_values.items():
                if v > -1e8: # Threshold to filter out masked values
                    valid_q[k] = v
        
        q1_str, q2_str, q3_str = "-", "-", "-"
        if valid_q:
            # Sort by value descending
            top_q = sorted(valid_q.items(), key=lambda item: item[1], reverse=True)[:3]
            
            if len(top_q) > 0:
                q1_str = f"{top_q[0][0]}: {top_q[0][1]:.3f}"
            if len(top_q) > 1:
                q2_str = f"{top_q[1][0]}: {top_q[1][1]:.3f}"
            if len(top_q) > 2:
                q3_str = f"{top_q[2][0]}: {top_q[2][1]:.3f}"
        
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
            gold_val = np.exp(float(decoded['Global']['Gold_Log'])) - 1
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
            q1_str,
            q2_str,
            q3_str,
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

    def commit_table(self, table_name="Episode_Trace"):
        """
        将缓冲区的内容提交为 Wandb Table。通常在一个 Episode 结束时调用。
        """
        if not self.enabled:
            return
            
        if not self.step_buffer:
            print("[WandbLogger] Step buffer is empty, nothing to commit.")
            return
            
        print(f"[WandbLogger] Committing table '{table_name}' with {len(self.step_buffer)} rows...")
        table = wandb.Table(columns=self.columns, data=self.step_buffer)
        wandb.log({table_name: table})
        self.step_buffer = [] # Clear buffer

    def finish(self):
        if not self.enabled:
            return

        # 1. Try normal finish
        try:
            self.run.finish()
        except Exception as e:
            print(f"[WandbLogger] Run finish failed (Network issue?): {e}")
        
        # 2. Check if we need to force sync via CLI
        # This is useful if the run was offline OR if the online finish failed
        try:
            # Check if offline mode was active
            is_offline = os.environ.get("WANDB_MODE") == "offline" or (self.run and self.run.settings.mode == "offline")
            
            if is_offline:
                print("[WandbLogger] Run was OFFLINE. Attempting to force sync via CLI now...")
                if self.run and self.run.dir:
                    # self.run.dir usually points to .../files. We need the run directory (parent)
                    # e.g. wandb/run-2025.../files -> wandb/run-2025...
                    run_dir = os.path.dirname(self.run.dir)
                    if os.path.exists(run_dir):
                        print(f"[WandbLogger] Syncing directory: {run_dir}")
                        subprocess.run(["wandb", "sync", run_dir], check=False)
                    else:
                        print(f"[WandbLogger] Run directory not found: {run_dir}")
        except Exception as e:
            print(f"[WandbLogger] Auto-sync failed: {e}")
            print("You may need to manually run 'wandb sync wandb/latest-run'")
