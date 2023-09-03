import json
import os
import numpy as np
from collections import OrderedDict

import rlcard
from rlcard.envs import Env
from rlcard.games.limitholdem import Game

DEFAULT_GAME_CONFIG = {
        'game_num_players': 2,
        }

class LimitholdemEnv(Env):
    ''' Limitholdem Environment
    '''

    def __init__(self, config):
        ''' Initialize the Limitholdem environment
        '''
        self.name = 'limit-holdem'
        self.default_game_config = DEFAULT_GAME_CONFIG
        self.game = Game()
        super().__init__(config)
        self.actions = ['call', 'raise', 'fold', 'check']
        self.state_shape = [[72] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

        self.card_state = [np.zeros((6, 4, 13)) for _ in range(self.num_players)]
        self.action_state = np.zeros((24, 3, 4))
        self.cont_actions = 0
        self.last_round = 0
        self.previous_legal_actions=None


        with open(os.path.join(rlcard.__path__[0], 'games/limitholdem/card2index.json'), 'r') as file:
            self.card2index = json.load(file)

    def _get_legal_actions(self):
        ''' Get all legal actions

        Returns:
            encoded_action_list (list): return encoded legal action list (from str to int)
        '''
        return self.game.get_legal_actions()

    def _extract_state(self, state):
        ''' Extract the state representation from state dictionary for agent

        Note: 1 tensor for cards and 1 tensor for actions.

        Args:
            state (dict): Original state from the game

        Returns:
            observation (list): combine the player's score and dealer's observable score for observation
        '''
        card_tensor = np.zeros((6, 4, 13))
        action_tensor = self.action_state.copy()
        extracted_state = {}

        # Handling the card tensor
        hand = state['hand']
        public_cards = state['public_cards']
        cards = public_cards + hand

        for card in hand:
            idx = self.card2index[card]
            suit = int(idx / 13)
            rank = idx % 13
            card_tensor[0][suit][rank] = 1

        if public_cards:
            i = 0
            for card in public_cards:
                i += 1
                if i <= 3:
                    z = 1
                elif i == 4:
                    z = 2
                elif i == 5:
                    z = 3
                idx = self.card2index[card]
                suit = int(idx / 13)
                rank = idx % 13
                card_tensor[z][suit][rank] = 1
                card_tensor[4][suit][rank] = 1

        for card in cards:
            idx = self.card2index[card]
            suit = int(idx / 13)
            rank = idx % 13
            card_tensor[5][suit][rank] = 1

        self.card_state = card_tensor[self.game.game_pointer]

        # Handling the action tensor
        round = self.game.round_counter

        z1 = self.last_round * 6 + self.cont_actions

        """action_tensor[action_number x round][player//legal][action]"""
        a = len(self.action_recorder)
        if a > 0 and self.last_round < 4:
            # get action number
            # get legal actions
            for i in self.previous_legal_actions:
                action_tensor[z1][2][i] = 1

            last_action = self.action_recorder[a - 1][1]
            player = self.action_recorder[a - 1][0]
            action_tensor[z1][player][self.actions.index(last_action)] = 1
            self.cont_actions += 1

            if round != self.last_round:
                self.last_round = round
                self.cont_actions = 0

        self.action_state = action_tensor

        legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
        self.previous_legal_actions = legal_actions

        extracted_state['card_tensor'] = card_tensor
        extracted_state['action_tensor'] = action_tensor
        extracted_state['raw_obs'] = state
        extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
        extracted_state['action_record'] = self.action_recorder
        extracted_state['legal_actions'] = legal_actions

        return extracted_state


    # def _extract_state(self, state):
    #     ''' Extract the state representation from state dictionary for agent
    #
    #     Note: Currently the use the hand cards and the public cards.
    #
    #     Args:
    #         state (dict): Original state from the game
    #
    #     Returns:
    #         observation (list): combine the player's score and dealer's observable score for observation
    #     '''
    #     extracted_state = {}
    #
    #     legal_actions = OrderedDict({self.actions.index(a): None for a in state['legal_actions']})
    #
    #
    #     public_cards = state['public_cards']
    #     hand = state['hand']
    #     raise_nums = state['raise_nums']
    #     cards = public_cards + hand
    #     idx = [self.card2index[card] for card in cards]
    #     obs = np.zeros(72)
    #     obs[idx] = 1
    #     for i, num in enumerate(raise_nums):
    #         obs[52 + i * 5 + num] = 1
    #
    #     """"""
    #     card_tensor = np.zeros((6, 4, 13))
    #     action_tensor = self.action_state.copy()
    #     extracted_state = {}
    #
    #     # Handling the card tensor
    #     hand = state['hand']
    #     public_cards = state['public_cards']
    #     cards = public_cards + hand
    #
    #     for card in hand:
    #         idx = self.card2index[card]
    #         suit = int(idx / 13)
    #         rank = idx % 13
    #         card_tensor[0][suit][rank] = 1
    #
    #     if public_cards:
    #         i = 0
    #         for card in public_cards:
    #             i += 1
    #             if i <= 3:
    #                 z = 1
    #             elif i == 4:
    #                 z = 2
    #             elif i == 5:
    #                 z = 3
    #             idx = self.card2index[card]
    #             suit = int(idx / 13)
    #             rank = idx % 13
    #             card_tensor[z][suit][rank] = 1
    #             card_tensor[4][suit][rank] = 1
    #
    #     for card in cards:
    #         idx = self.card2index[card]
    #         suit = int(idx / 13)
    #         rank = idx % 13
    #         card_tensor[5][suit][rank] = 1
    #
    #     self.card_state = card_tensor[self.game.game_pointer]
    #
    #     # Handling the action tensor
    #     round = self.game.round_counter
    #
    #     z1 = self.last_round * 6 + self.cont_actions
    #
    #
    #
    #
    #
    #
    #     """action_tensor[action_number x round][player//legal][action]"""
    #     a = len(self.action_recorder)
    #     if a > 0 and self.last_round < 4:
    #         # get action number
    #         # get legal actions
    #         for i in self.previous_legal_actions:
    #             action_tensor[z1][2][i] = 1
    #
    #         last_action = self.action_recorder[a - 1][1]
    #         player = self.action_recorder[a - 1][0]
    #         action_tensor[z1][player][self.actions.index(last_action)] = 1
    #         self.cont_actions += 1
    #
    #         if round != self.last_round:
    #             self.last_round = round
    #             self.cont_actions = 0
    #
    #
    #     self.action_state = action_tensor
    #     self.previous_legal_actions = legal_actions
    #     print("======================================================================")
    #     print(self.action_state)
    #     print("======================================================================")
    #     """"""
    #     extracted_state['obs'] = obs
    #
    #     extracted_state['raw_obs'] = state
    #     extracted_state['raw_legal_actions'] = [a for a in state['legal_actions']]
    #     extracted_state['action_record'] = self.action_recorder
    #     extracted_state['legal_actions'] = legal_actions
    #     # print(public_cards)
    #     # print("Actions--------------------------------")
    #     # print(self.action_recorder)
    #     # print("ddddddd")
    #     # a = len(self.action_recorder)
    #     # if a > 0: print(self.action_recorder[a-1][1])
    #     # print(legal_actions)
    #     return extracted_state



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
