import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.newlimitholdem import Game

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class NewLimitholdemEnv(Env):
    ''' NewLimitholdem Environment
    '''

    def __init__(self, config):
        ''' Initialize the New Limitholdem environment
        '''
        self.name = 'new-limit-holdem'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = ['call', 'raise', 'fold', 'check']
        self.state_shape = [[32] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        with open(os.path.join(rlcard.__path__[0], 'games/newlimitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        ''' Get all legal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: Currently the use the hand cards and the public cards.
        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        extracted_state = {}

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        extracted_state['legal_actions'] = legal_actions

        public_cards = state['public_cards']
        hand = state['hand']

        obs = np.zeros(32)
        idx = [self.card2index[card] for card in hand]
        obs[idx] = 1
        idx2 = []
        for j, card in enumerate(public_cards):
            idx2 = [(self.card2index[card] + 5*j)]
            obs[idx2] = 1
        obs[state['my_chips'] + 15] = 1
        obs[sum(state['all_chips']) - state['my_chips'] + 21] = 1
        for i in legal_actions:
            idx3 = [i + 27]
            obs[idx3] = 1
        obs[31] = state['first']

        extracted_state['obs'] = obs
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        extracted_state['obs'] = obs
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder

        return extracted_state

    def get_payoffs(self):
        ''' Get the payoff of a game

        Returns:
           payoffs (list): list of payoffs
        '''
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        ''' Decode the action for applying to the game

        Args:
            action id (int): action id

        Returns:
            action (str): action for the game
        '''
        legal_actions = self.game.get_legal_actions()
        if self.actions[action_id] not in legal_actions:
            if 'check' in legal_actions:
                return 'check'
            else:
                return 'fold'
        return self.actions[action_id]

    def get_perfect_information(self):
        ''' Get the perfect information of the current state

        Returns:
            (dict): A dictionary of all the perfect information of the current state
        '''
        state = {}
        state['chips'] = [self.game.players[i].in_chips for i in range(self.num_players)]
        state['public_card'] = [c.get_index() for c in self.game.public_cards] if self.game.public_cards else None
        state['hand_cards'] = [[c.get_index() for c in self.game.players[i].hand] for i in range(self.num_players)]
        state['current_player'] = self.game.game_pointer
        state['legal_actions'] = self.game.get_legal_actions()
        return state

    def get_agents(self):
        return self.agents

    def change_public_cards(self, card1, card2):
        self.game.change_public_cards(card1, card2)