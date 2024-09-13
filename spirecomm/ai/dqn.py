from spirecomm.communication.action import *
import random
class DQN:
    def __init__(self):

        pass
    def generate_command_space(self, game_state):
        all_choices = []
        commands = game_state.available_commands
        # 将commands数组中的'key','click','state','wait'删掉
        commands = [command for command in commands if command not in ['key', 'click', 'state', 'wait']]
        for command in commands:
            if command in ['end', 'proceed', 'return']:
                all_choices.append(Action(command))
            elif command == 'play':
                for i in range(len(game_state.hand)):
                    if game_state.hand[i].is_playable:
                        if game_state.hand[i].has_target:
                            for j in range(len(game_state.monsters)):
                                all_choices.append(Action(command + ' ' + str(i+1) + ' ' + str(j)))
                                with open('log.txt', 'a') as f:
                                    f.write('-------play-------\n')
                                    f.write(command + ' ' + str(i+1) + ' ' + str(j) + '\n')
                        else:
                            all_choices.append(Action(command + ' ' + str(i+1)))
            elif command == 'potion':
                potions = game_state.potions
                for i in range(len(potions)):
                    if potions[i].can_discard:
                        all_choices.append(Action(command + ' discard ' + str(i)))
                    if potions[i].can_use:
                        if potions[i].requires_target:
                            for j in range(len(game_state.monsters)):
                                all_choices.append(Action(command + ' use ' + str(i) + ' ' + str(j)))
                        else:
                            all_choices.append(Action(command + ' use ' + str(i)))
            elif command == 'choose':
                if game_state.choice_available:
                    for i in range(len(game_state.choice_list)):
                        all_choices.append(Action(command + ' ' + str(i)))
        return all_choices
    def handle_error(self, error):
        raise Exception(error)
    def get_next_action_in_game(self, game_state):
        all_choices = self.generate_command_space(game_state)
        return random.choice(all_choices)
    def change_class(self, new_class):
        self.chosen_class = new_class
    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)