import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from spirecomm.utils.path import get_root_dir


WINDOW_SIZE = 40
MIN_EPISODES_BEFORE_PROMOTION = 12
ACT2_FLOOR_THRESHOLD = 18
ACT3_FLOOR_THRESHOLD = 35
ACT4_FLOOR_THRESHOLD = 52


STAGE_REWARD_CONFIGS: Dict[int, Dict[str, float]] = {
    0: {
        "DAMAGE_DEALT_MULTIPLIER": 0.4,
        "DAMAGE_TAKEN_MULTIPLIER": -0.9,
        "MONSTER_DEATH_REWARD": 6.0,
        "WIN_BATTLE_REWARD": 10.0,
        "WIN_ELITE_MONSTER_BONUS": 12.0,
        "GOLD_GAINED_REWARD": 0.2,
        "POTION_GAINED_REWARD": 2.0,
        "POTION_KEEP_BONUS": 2.0,
        "POTION_USE_PENALTY": -8.0,
        "FLOOR_INCREASE_ACT1": 6.0,
        "FLOOR_INCREASE_ACT2": 7.0,
        "FLOOR_INCREASE_ACT3": 8.0,
        "FLOOR_INCREASE_ACT4": 9.0,
        "WIN_ACT1_BOSS_BONUS": 10.0,
        "WIN_ACT2_BOSS_BONUS": 10.0,
        "WIN_ACT3_BOSS_BONUS": 10.0,
        "WIN_FINAL_BOSS_REWARD": 10.0,
        "LOSE_BATTLE_REWARD": -10.0,
    },
    1: {
        "DAMAGE_DEALT_MULTIPLIER": 0.28,
        "DAMAGE_TAKEN_MULTIPLIER": -0.95,
        "MONSTER_DEATH_REWARD": 4.0,
        "WIN_BATTLE_REWARD": 7.0,
        "WIN_ELITE_MONSTER_BONUS": 10.0,
        "GOLD_GAINED_REWARD": 0.1,
        "POTION_GAINED_REWARD": 1.0,
        "POTION_KEEP_BONUS": 1.5,
        "POTION_USE_PENALTY": -6.0,
        "FLOOR_INCREASE_ACT1": 7.5,
        "FLOOR_INCREASE_ACT2": 8.5,
        "FLOOR_INCREASE_ACT3": 9.5,
        "FLOOR_INCREASE_ACT4": 10.0,
        "WIN_ACT1_BOSS_BONUS": 10.0,
        "WIN_ACT2_BOSS_BONUS": 10.0,
        "WIN_ACT3_BOSS_BONUS": 10.0,
        "WIN_FINAL_BOSS_REWARD": 10.0,
        "LOSE_BATTLE_REWARD": -10.0,
    },
    2: {
        "DAMAGE_DEALT_MULTIPLIER": 0.16,
        "DAMAGE_TAKEN_MULTIPLIER": -1.0,
        "MONSTER_DEATH_REWARD": 2.0,
        "WIN_BATTLE_REWARD": 4.0,
        "WIN_ELITE_MONSTER_BONUS": 6.0,
        "GOLD_GAINED_REWARD": 0.05,
        "POTION_GAINED_REWARD": 0.5,
        "POTION_KEEP_BONUS": 0.5,
        "POTION_USE_PENALTY": -4.0,
        "FLOOR_INCREASE_ACT1": 8.5,
        "FLOOR_INCREASE_ACT2": 9.5,
        "FLOOR_INCREASE_ACT3": 10.0,
        "FLOOR_INCREASE_ACT4": 10.0,
        "WIN_ACT1_BOSS_BONUS": 10.0,
        "WIN_ACT2_BOSS_BONUS": 10.0,
        "WIN_ACT3_BOSS_BONUS": 10.0,
        "WIN_FINAL_BOSS_REWARD": 10.0,
        "LOSE_BATTLE_REWARD": -10.0,
    },
    3: {
        "DAMAGE_DEALT_MULTIPLIER": 0.08,
        "DAMAGE_TAKEN_MULTIPLIER": -1.0,
        "MONSTER_DEATH_REWARD": 1.0,
        "WIN_BATTLE_REWARD": 2.0,
        "WIN_ELITE_MONSTER_BONUS": 3.0,
        "GOLD_GAINED_REWARD": 0.0,
        "POTION_GAINED_REWARD": 0.0,
        "POTION_KEEP_BONUS": 0.0,
        "POTION_USE_PENALTY": -2.0,
        "FLOOR_INCREASE_ACT1": 9.0,
        "FLOOR_INCREASE_ACT2": 10.0,
        "FLOOR_INCREASE_ACT3": 10.0,
        "FLOOR_INCREASE_ACT4": 10.0,
        "WIN_ACT1_BOSS_BONUS": 10.0,
        "WIN_ACT2_BOSS_BONUS": 10.0,
        "WIN_ACT3_BOSS_BONUS": 10.0,
        "WIN_FINAL_BOSS_REWARD": 10.0,
        "LOSE_BATTLE_REWARD": -10.0,
    },
}


@dataclass
class RewardSchedulerUpdate:
    current_stage: int
    next_stage: int
    changed: bool
    metrics: Dict[str, float]
    reward_config: Dict[str, float]


class RewardAutoScheduler:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = root_dir or get_root_dir()
        self.state_path = os.path.join(self.root_dir, "data", "reward_scheduler_state.json")
        self.dynamic_config_path = os.path.join(self.root_dir, "dynamic_config.json")
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.state = self._load_state()

    def initialize(self) -> RewardSchedulerUpdate:
        stage = int(self.state.get("current_stage", 0))
        metrics = self._compute_metrics(self.state.get("history", []))
        reward_config = self._write_dynamic_config(stage, metrics)
        self._save_state()
        return RewardSchedulerUpdate(
            current_stage=stage,
            next_stage=stage,
            changed=False,
            metrics=metrics,
            reward_config=reward_config,
        )

    def record_episode(self, floor_reached: int, victory: bool, player_class: Optional[str] = None) -> RewardSchedulerUpdate:
        history = list(self.state.get("history", []))
        history.append(
            {
                "floor": int(floor_reached if floor_reached is not None else 0),
                "victory": bool(victory),
                "player_class": player_class or "",
            }
        )
        history = history[-WINDOW_SIZE:]
        self.state["history"] = history

        current_stage = int(self.state.get("current_stage", 0))
        metrics = self._compute_metrics(history)
        target_stage = self._determine_target_stage(history, metrics)
        next_stage = max(current_stage, target_stage)
        self.state["current_stage"] = next_stage

        reward_config = self._write_dynamic_config(next_stage, metrics)
        self._save_state()
        return RewardSchedulerUpdate(
            current_stage=current_stage,
            next_stage=next_stage,
            changed=next_stage != current_stage,
            metrics=metrics,
            reward_config=reward_config,
        )

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.state_path):
            return {"current_stage": 0, "history": []}
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("scheduler state is not a dict")
            data.setdefault("current_stage", 0)
            data.setdefault("history", [])
            return data
        except Exception:
            return {"current_stage": 0, "history": []}

    def _save_state(self) -> None:
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.state_path)

    def _load_dynamic_config(self) -> Dict[str, Any]:
        if not os.path.exists(self.dynamic_config_path):
            return {}
        try:
            with open(self.dynamic_config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_dynamic_config(self, stage: int, metrics: Dict[str, float]) -> Dict[str, float]:
        data = self._load_dynamic_config()
        reward = data.get("reward", {})
        if not isinstance(reward, dict):
            reward = {}

        stage_reward = dict(STAGE_REWARD_CONFIGS[stage])
        reward.update(stage_reward)
        data["reward"] = reward
        data["reward_scheduler"] = {
            "enabled": True,
            "current_stage": int(stage),
            "window_size": WINDOW_SIZE,
            "metrics": metrics,
        }

        tmp_path = self.dynamic_config_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.dynamic_config_path)
        return stage_reward

    def _compute_metrics(self, history: List[Dict[str, object]]) -> Dict[str, float]:
        if not history:
            return {
                "episodes": 0.0,
                "avg_floor": 0.0,
                "best_floor": 0.0,
                "victory_rate": 0.0,
                "act2_reach_rate": 0.0,
                "act3_reach_rate": 0.0,
                "act4_reach_rate": 0.0,
            }

        floors = [int(item.get("floor", 0)) for item in history]
        victories = [1.0 if item.get("victory") else 0.0 for item in history]
        count = float(len(history))

        return {
            "episodes": count,
            "avg_floor": sum(floors) / count,
            "best_floor": float(max(floors)),
            "victory_rate": sum(victories) / count,
            "act2_reach_rate": sum(1.0 for floor in floors if floor >= ACT2_FLOOR_THRESHOLD) / count,
            "act3_reach_rate": sum(1.0 for floor in floors if floor >= ACT3_FLOOR_THRESHOLD) / count,
            "act4_reach_rate": sum(1.0 for floor in floors if floor >= ACT4_FLOOR_THRESHOLD) / count,
        }

    def _determine_target_stage(self, history: List[Dict[str, object]], metrics: Dict[str, float]) -> int:
        if len(history) < MIN_EPISODES_BEFORE_PROMOTION:
            return 0

        if metrics["victory_rate"] > 0.0 or metrics["avg_floor"] >= 40.0 or metrics["act4_reach_rate"] >= 0.10:
            return 3
        if metrics["avg_floor"] >= 22.0 or metrics["act3_reach_rate"] >= 0.20:
            return 2
        if metrics["avg_floor"] >= 12.0 or metrics["act2_reach_rate"] >= 0.30:
            return 1
        return 0
