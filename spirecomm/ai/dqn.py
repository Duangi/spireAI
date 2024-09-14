from spirecomm.communication.action import *
from spirecomm.spire.game import Game
import random
import torch 
class DQN:
    def __init__(self):
        self.game = Game()
    def generate_action_space(self, game_state):

        self.game = game_state
        commands = game_state.available_commands
        # 将commands数组中的'key','click','state','wait'删掉
        commands = [command for command in commands if command not in ['key', 'click', 'state', 'wait']]
        # 补丁：当在休息站时，选择了升级，但是在选择卡牌时，没有选择卡牌，直接选择了取消，导致收不到新的available_commands，导致无法继续游戏
        # 补丁2：在给精灵献卡时也会出现类似的问题。
        # 如果同时存在cancel和confirm，那么删掉cancel
        if 'cancel' in commands and 'confirm' in commands:
            commands.remove('cancel')
        all_choices = []
        for command in commands:
            command = command.lower()
            if command not in ['play', 'potion', 'choose']:
                all_choices.append(Action(command))
            elif command == 'play':
                for i in range(len(game_state.hand)):
                    if game_state.hand[i].is_playable:
                        if game_state.hand[i].has_target:
                            available_monsters = [monster for monster in game_state.monsters if monster.current_hp > 0 and not monster.half_dead and not monster.is_gone]
                            for monster in available_monsters:
                                all_choices.append(PlayCardAction(card=game_state.hand[i], target_monster=monster))
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
        all_choices = self.generate_action_space(game_state)
        return random.choice(all_choices)
    
    def change_class(self, new_class):
        self.chosen_class = new_class
    def get_next_action_out_of_game(self):
        return StartGameAction(self.chosen_class)