import sys
import os
import torch
import wandb

if __name__ == "__main__":
    # Add project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from spirecomm.ai.dqn_core.algorithm import SpireAgent
    from spirecomm.ai.dqn_core.model import SpireConfig, SpireState
    from spirecomm.ai.dqn_core.wandb_logger import WandbLogger
    from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases
    from spirecomm.ai.dqn_core.state import GameStateProcessor
    from spirecomm.spire.game import Game
    from dataclasses import fields
    from itertools import cycle

    print("Initializing WandbLogger...")
    # Logger will check .env for WANDB_API_KEY. If missing, it will disable itself.
    # Note: WandbLogger has built-in fallback to offline mode if network fails.
    logger = WandbLogger(project_name="spire-ai-test", run_name="test_run", config={"test": True})

    print("Initializing SpireAgent with Logger...")
    config = SpireConfig()
    # Prioritize GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    agent = SpireAgent(config, device=device, wandb_logger=logger)

    processor = GameStateProcessor()
    
    print("Starting Test Loop over multiple cases...")
    
    # 1. Filter for valid game states first
    print("Filtering test cases for valid game states...")
    valid_games = []
    
    # Iterate through ALL test cases to find valid ones
    for i, game_json in enumerate(test_cases):
        if not isinstance(game_json, dict):
            continue
            
        try:
            # Logic adapted from model_test_runner.py to ensure we correctly parse the state
            # and force in_game=True if missing, so we don't skip valid test cases.
            if 'game_state' not in game_json:
                case_wrapper = {
                    'game_state': game_json,
                    'available_commands': game_json.get('available_commands', []),
                    'in_game': game_json.get('in_game', True) # Default to True for test cases
                }
                game = Game.from_json(case_wrapper)
            else:
                game = Game.from_json(game_json)

            # Check if we can generate a tensor from it
            state = processor.get_state_tensor(game)
            if isinstance(state, SpireState):
                valid_games.append(game)
                
        except Exception as e:
            # print(f"Skipping #{i} due to error: {e}")
            continue
            
    print(f"Found {len(valid_games)} valid game states. Starting training loop...")

    if not valid_games:
        print("No valid game states found in test_cases!")
        sys.exit(1)

    # 2. Cycle through valid states to simulate training
    count = 0
    # Use cycle to ensure we get enough steps even if we only found a few valid states
    for game in cycle(valid_games):
        if count >= 100: # Run 100 steps to ensure we trigger training multiple times (batch_size=32)
            break

        try:
            # print(f"Processing Step #{count+1}...")
            
            state = processor.get_state_tensor(game)
            
            # 1. Choose Action
            # SpireAgent.choose_action takes (state, game_state_obj) and handles batching/device internally
            action = agent.choose_action(state, game)
            # print(f"  -> Chosen Action: {action}")
            
            # 2. Remember (Log Step)
            # Use same state as next_state for dummy transition
            next_state = state 
            reward = 0.1 * (count % 10) # Fake reward variation
            reward_details = f"FakeReward: {reward:.1f}"
            done = False
            
            # Format action string nicely for test runner
            if hasattr(action, 'to_string'):
                action_str = action.to_string()
            else:
                action_str = str(action)

            # Add extra info for PlayAction to be more readable (e.g. card name)
            if hasattr(action, 'hand_idx') and hasattr(game, 'hand'):
                try:
                    if hasattr(action, 'decomposed_type') and action.decomposed_type.name == 'PLAY':
                        card_name = game.hand[action.hand_idx].name
                        action_str += f" ({card_name})"
                except:
                    pass

            # Pass the original action object to remember, let remember handle logging string conversion
            # But wait, remember expects action object for replay buffer, but logs string.
            # My previous edit to algorithm.py handles conversion inside remember.
            # So here we just pass action object.
            agent.remember(state, action, reward, next_state, done, reward_details)
            
            # 3. Train (Log Metrics) occasionally
            # Need enough memory to train (batch_size=32)
            if len(agent.memory) >= 32:
                agent.train()
                if count % 10 == 0:
                    print(f"  -> Step {count}: Training step executed (Metrics Logged).")
            
            count += 1
            
        except Exception as e:
            print(f"  -> Skipped Step #{count+1} due to error: {e}")
            continue

    # Force commit the table at the end
    if logger.enabled:
        try:
            logger.commit_table()
            logger.finish()
        except Exception as e:
            print(f"Wandb logging finish failed (likely network issue): {e}")
            print("Logs should still be available locally in the 'wandb' directory.")
            
    print("Wandb Run Finished.")
