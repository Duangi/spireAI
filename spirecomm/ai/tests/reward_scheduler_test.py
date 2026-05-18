import json

from spirecomm.ai.reward_scheduler import RewardAutoScheduler


def test_reward_scheduler_promotes_stage_and_updates_dynamic_config(tmp_path):
    dynamic_config_path = tmp_path / "dynamic_config.json"
    dynamic_config_path.write_text(
        json.dumps({"exploration_total_steps": 300000, "reward": {"WASTE_ENERGY_PENALTY": -15.0}}),
        encoding="utf-8",
    )

    scheduler = RewardAutoScheduler(root_dir=str(tmp_path))
    init_result = scheduler.initialize()

    assert init_result.next_stage == 0
    assert init_result.reward_config["WIN_BATTLE_REWARD"] == 10.0

    for _ in range(12):
        result = scheduler.record_episode(floor_reached=18, victory=False, player_class="IRONCLAD")

    assert result.changed is True
    assert result.current_stage == 0
    assert result.next_stage == 1
    assert result.metrics["avg_floor"] == 18.0

    data = json.loads(dynamic_config_path.read_text(encoding="utf-8"))
    assert data["exploration_total_steps"] == 300000
    assert data["reward"]["WASTE_ENERGY_PENALTY"] == -15.0
    assert data["reward"]["WIN_BATTLE_REWARD"] == 7.0
    assert data["reward_scheduler"]["current_stage"] == 1


def test_reward_scheduler_stage_never_regresses(tmp_path):
    scheduler = RewardAutoScheduler(root_dir=str(tmp_path))
    scheduler.initialize()

    for _ in range(12):
        scheduler.record_episode(floor_reached=35, victory=False)

    assert scheduler.state["current_stage"] == 2

    for _ in range(20):
        result = scheduler.record_episode(floor_reached=3, victory=False)

    assert result.next_stage == 2
    assert scheduler.state["current_stage"] == 2
