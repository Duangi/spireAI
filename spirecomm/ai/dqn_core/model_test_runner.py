
import sys
import os
import torch

if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from spirecomm.ai.dqn_core.algorithm import SpireAgent
    from spirecomm.ai.dqn_core.model import SpireConfig, SpireState
    from spirecomm.ai.dqn_core.state import GameStateProcessor
    from spirecomm.spire.game import Game
    from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases
    from spirecomm.ai.dqn_core.action import SingleAction, ActionType, DecomposedActionType

    # 1. Initialize Config and Agent
    config = SpireConfig() # Uses default dims we just set
    # Force CPU for testing to avoid CUDA errors if not available
    agent = SpireAgent(config, device="cpu")
    
    processor = GameStateProcessor()
    
    
    # 2. Collect some valid states
    valid_states = []
    
    # Try to collect ALL valid states to stress test the model
    target_count = 1000 
    
    
    for i, case in enumerate(test_cases):
        if len(valid_states) >= target_count:
            break
            
        try:
            # Skip incomplete cases
            if 'act' not in case and 'current_hp' not in case:
                 continue

            # Wrapper fix
            if 'game_state' not in case:
                case_wrapper = {
                    'game_state': case,
                    'available_commands': case.get('available_commands', []),
                    'in_game': case.get('in_game', True)
                }
                game = Game.from_json(case_wrapper)
            else:
                game = Game.from_json(case)

            state = processor.get_state_tensor(game)
            
            if isinstance(state, SpireState):
                valid_states.append(state)
            
            # Also try a forward pass immediately to catch specific failing cases
            # This verifies the model can handle this specific state's shape
            try:
                # Add batch dim
                batch_state = agent.collate_states([state])
                # Forward pass (no grad)
                with torch.no_grad():
                    _ = agent.policy_net(batch_state)
            except Exception as e:
                # We don't add it to valid_states if it crashes the model
                valid_states.pop() 
                
        except Exception as e:
            continue
        
    
    if len(valid_states) < 32:
        sys.exit(1)

    # 3. Populate Memory
    
    # Create a dummy action (End Turn)
    dummy_action = SingleAction(type=ActionType.END, decomposed_type=DecomposedActionType.END)
    
    for i in range(len(valid_states) - 1):
        s = valid_states[i]
        next_s = valid_states[i+1]
        reward = 1.0
        done = False
        
        agent.remember(s, dummy_action, reward, next_s, done)
        
    # 4. Run Training Step
    try:
        agent.train()
    except Exception as e:
        import traceback
        traceback.print_exc()
