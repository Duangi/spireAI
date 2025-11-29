import pytest
import numpy as np
from unittest.mock import Mock, patch
from spirecomm.spire.game import Game
from spirecomm.spire.card import Card
from dqn_core import DQNAgent, DQN, GameStateProcessor, CardManager

# -------------------- 辅助函数：模拟游戏状态 --------------------
def mock_game_state():
    """生成模拟的Game对象（用于测试）"""
    game = Mock(spec=Game)
    # 模拟手牌（3张可使用的卡牌）
    mock_card = Mock(spec=Card)
    mock_card.is_playable = True
    mock_card.has_target = False
    game.hand = [mock_card, mock_card, mock_card]
    # 模拟怪物（2个存活）
    monster1 = Mock()
    monster1.current_hp = 100
    monster1.half_dead = False
    monster1.is_gone = False
    monster2 = Mock()
    monster2.current_hp = 100
    monster2.half_dead = False
    monster2.is_gone = False
    game.monsters = [monster1, monster2]
    # 模拟可用命令
    game.available_commands = ["play", "potion", "confirm", "cancel"]
    # 模拟其他属性（hp、floor等）
    game.current_hp = 100
    game.max_hp = 200
    game.floor = 5
    game.act = 1
    game.gold = 500
    game.ascension_level = 10
    game.character = Mock(value=0)  # 假设角色类型为0
    return game

# -------------------- 测试GameStateProcessor --------------------
def test_game_state_processor_hand_vector():
    """测试手牌向量预处理"""
    card_manager = CardManager()  # 假设CardManager已正确实现
    processor = GameStateProcessor(card_manager)
    game = mock_game_state()
    
    hand_vector = processor.get_hand_vector(game)
    # 验证形状：(10, 25)（max_hand_size=10，每张卡牌25维）
    assert hand_vector.shape == (10, 25)
    # 验证前3张为真实卡牌向量（非零），后7张为零向量
    assert not np.allclose(hand_vector[:3], 0)
    assert np.allclose(hand_vector[3:], 0)

def test_game_state_processor_fixed_vector():
    """测试定长数值向量预处理"""
    card_manager = CardManager()
    processor = GameStateProcessor(card_manager)
    game = mock_game_state()
    
    fixed_vector = processor.get_fixed_vector(game)
    # 验证维度：7（数值） + 5（角色独热）= 12维
    assert len(fixed_vector) == 12
    # 验证归一化后的值在0-1范围内
    assert 0 <= fixed_vector[0] <= 1  # current_hp/200
    assert 0 <= fixed_vector[1] <= 1  # max_hp/200
    assert fixed_vector[10] == 1      # 角色类型独热编码（假设value=0）

# -------------------- 测试DQNAgent --------------------
def test_dqn_agent_generate_action_space():
    """测试动作空间生成逻辑"""
    agent = DQNAgent(use_dqn=False)
    game = mock_game_state()
    game.available_commands = ["play", "potion", "confirm", "cancel"]  # 包含cancel和confirm
    
    actions = agent.generate_action_space(game)
    # 验证过滤了'cancel'（因为同时存在confirm）
    assert "cancel" not in [str(action) for action in actions]
    # 验证生成了play、potion、confirm的动作
    assert any("play" in str(action) for action in actions)
    assert any("potion" in str(action) for action in actions)
    assert any("confirm" in str(action) for action in actions)

# -------------------- 测试DQN类 --------------------
def test_dqn_act():
    """测试DQN的动作选择逻辑"""
    card_manager = CardManager()
    processor = GameStateProcessor(card_manager)
    dqn = DQN(action_size=10, state_processor=processor)  # 假设action_size=10（commands.json有10个命令）
    game = mock_game_state()
    
    # 预处理状态
    fixed_vector = processor.get_fixed_vector(game)
    hand_vector = processor.get_hand_vector(game)
    state = [np.array([fixed_vector]), np.array([hand_vector])]
    
    # 测试随机策略（epsilon=1.0时应返回随机动作）
    action_idx = dqn.act(state)
    assert 0 <= action_idx < 10  # 动作索引在0-9范围内
    
    # 测试贪心策略（强制epsilon=0）
    dqn.epsilon = 0.0
    action_idx = dqn.act(state)
    assert 0 <= action_idx < 10  # 应返回模型预测的最大Q值动作

def test_dqn_train():
    """测试DQN的训练逻辑（验证模型是否更新）"""
    card_manager = CardManager()
    processor = GameStateProcessor(card_manager)
    dqn = DQN(action_size=10, state_processor=processor)
    
    # 生成模拟经验（状态、动作、奖励、下一状态、done）
    state = [np.random.rand(1, 12), np.random.rand(1, 10, 25)]  # 固定向量12维，手牌向量(10,25)
    next_state = [np.random.rand(1, 12), np.random.rand(1, 10, 25)]
    dqn.remember(state, action=2, reward=1.0, next_state=next_state, done=False)
    
    # 训练前的模型输出
    initial_output = dqn.model.predict(state, verbose=0)
    
    # 执行训练（batch_size=1）
    dqn.train(batch_size=1)
    
    # 训练后的模型输出应变化（验证参数更新）
    post_train_output = dqn.model.predict(state, verbose=0)
    assert not np.allclose(initial_output, post_train_output)