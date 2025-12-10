
from spirecomm.ai.dqn_core.model import SpireState
from spirecomm.ai.dqn_core.state import GameStateProcessor
from spirecomm.spire.game import Game


if __name__ == "__main__":
    import sys
    import os
    # Add project root to path to allow imports
    # Assuming this file is at d:\Projects\spireAI\spirecomm\ai\dqn_core\state.py
    # We need to go up 3 levels to reach d:\Projects\spireAI
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    sys.path.append(project_root)
    
    from spirecomm.ai.tests.test_case.game_state_test_cases import test_cases
    
    processor = GameStateProcessor()
    
    print(f"Found {len(test_cases)} test cases.")
    
    success_count = 0
    skipped_count = 0
    
    for i, case in enumerate(test_cases):
        try:
            print(f"Testing case {i+1}...")
            
            # Check for Menu/Start states
            if case.get('in_game') is False:
                 print(f"Skipping case {i+1} (Menu/Start Screen - Not in game)")
                 skipped_count += 1
                 continue

            # Simple check to skip obviously non-game states (like the first one in the file)
            if 'act' not in case and 'current_hp' not in case:
                 print(f"Skipping case {i+1} (Incomplete State Data)")
                 skipped_count += 1
                 continue

            # Fix for Game.from_json expecting a wrapper with 'game_state'
            if 'game_state' not in case:
                # Construct a wrapper
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
                print(f"Case {i+1} passed! Generated SpireState.")
                success_count += 1
            else:
                print(f"Case {i+1} failed! Return type is {type(state)}")
                
        except Exception as e:
            print(f"Case {i+1} raised exception: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"Test finished. {success_count} cases passed. {skipped_count} cases skipped (Menu/Incomplete).")
