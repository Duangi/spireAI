# spireAI 运行逻辑整理

## 当前核心运行入口

- `/home/runner/work/spireAI/spireAI/worker.py`
  - 这是 Communication Mod 实际调用的运行入口。
  - 它负责启动游戏对局、驱动 `DQNAgent` 做决策，并把每一局的 transition 数据写到 `data/memory/`。
  - Worker 每局开始前会读取最新模型 `models/latest.pth`，并在开局前调用 `agent.reward_calculator.reload_config()`，所以 `dynamic_config.json` 里的 reward 参数会在这里生效。

- `/home/runner/work/spireAI/spireAI/trainer.py`
  - 这是离线训练入口。
  - 它负责扫描 `data/memory/` 和 `data/memory_remote/` 中的 `.pt` 数据文件，把 transition 放进 replay buffer，再执行训练。
  - Trainer 会周期性保存模型到 `models/step_*.pth`，同时刷新 `models/latest.pth` 给 worker 使用。

## 数据流

1. Mod 调用 `worker.py`
2. `worker.py` 跑一局游戏
3. 每局 transition 写入 `data/memory/<角色>/step_*.pt`
4. `trainer.py` 扫描这些 `.pt` 文件并训练
5. `trainer.py` 保存新模型到 `models/`
6. `worker.py` 下次开局前加载 `models/latest.pth`

## 奖励系统

- 奖励计算位置：
  - `/home/runner/work/spireAI/spireAI/spirecomm/ai/dqn_core/reward.py`
- 动态奖励配置入口：
  - `/home/runner/work/spireAI/spireAI/dynamic_config.json`
- 当前项目里 reward 会在 `worker.py` 侧生效，因为实际游戏 rollout 和经验采样都发生在 worker。

## 本次自动调奖方案

- 新增自动奖励调度器：
  - `/home/runner/work/spireAI/spireAI/spirecomm/ai/reward_scheduler.py`
- 调度依据：
  - 最近若干局的平均层数
  - 是否稳定到达 Act2 / Act3 / Act4
  - 是否已经通关
- 调度目标：
  - 训练早期保留更多战斗 shaping
  - 随着表现提升，逐步削弱金币/药水/普通战收益
  - 最后把重点收敛到更高层数、Boss、通关

## 后续改动前建议优先查看

- 本文件：`/home/runner/work/spireAI/spireAI/CLAUDE.md`
- `worker.py`
- `trainer.py`
- `spirecomm/ai/dqn_core/reward.py`
- `dynamic_config.json`
