import numpy as np
from rlcard.games.limitholdem.utils import Hand
import random


class BluffAgent(object):
    ''' A threshold agent. He will be playing limit-holdem.
        Will bet the maximum amount allowed in each round, provided that it has at least a high enough "combination".
        In round 1 it will always bet/raise with a K or Ace.
        In round 2 it will always bet/raise with any pair and three.
    '''

    def __init__(self, num_actions, env):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.bluff_threshold = 0.3
        self.env = env


    def step(self, state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        legal_actions = state['raw_legal_actions']
        lg = list(state['legal_actions'].keys())

        if len(state['raw_obs']['public_cards']) == 0:  #we are on 1st round i.e. no public cards
            hand = state['raw_obs']['hand']

            if (hand[0] in ['SA', 'HA', 'DA', 'CA'] and hand[1] in ['SA', 'HA', 'DA', 'CA']) or \
                    (hand[0] in ['SK', 'HK', 'DK', 'CK'] and hand[1] in ['SK', 'HK', 'DK', 'CK']):
                action = self.aggressive_actions(legal_actions)
                return action


            if (hand[0][1] == 'A' and hand[1][1] == 'K') or (hand[0][1] == 'K' and hand[1][1] == 'A'):

                action = self.aggressive_actions(legal_actions)
                return action


            # Check for connected high cards in late positions
            if (hand[0][1].isdigit() and hand[1][1].isdigit()) and \
                    (int(hand[0][1]) >= 9 and int(hand[1][1]) >= 9):
                action = self.aggressive_actions(legal_actions)
                return action

            if random.random() < self.bluff_threshold:
                action = self.aggressive_actions(legal_actions)
            else:
                action = self.normal_actions(legal_actions)
                return action
        else:   # second round raise only on pairs and threes
            cards = state['raw_obs']['hand'] + state['raw_obs']['public_cards']
            hand = Hand(cards)
            hand.sort_cards()
            hand.agent_cards()
            # straight flush
            if hand.has_straight_flush():
                action = self.aggressive_actions(legal_actions)
                return action

            # four of a kind
            if hand.has_four():
                action = self.aggressive_actions(legal_actions)
                return action

            # full house
            if hand.has_fullhouse():
                if self.env.game.round_counter == 1:
                    action = self.aggressive_actions(legal_actions)
                    return action
                else:
                    if random.random() < self.bluff_threshold:
                        action = self.aggressive_actions(legal_actions)
                        return action
                    else:
                        action = self.normal_actions(legal_actions)
                        return action

            if hand.has_flush():
                if random.random() < self.bluff_threshold:
                    action = self.aggressive_actions(legal_actions)
                    return action
                else:
                    action = self.normal_actions(legal_actions)
                    return action

            if hand.has_straight(hand.all_cards):
                if random.random() < self.bluff_threshold:
                    action = self.aggressive_actions(legal_actions)
                    return action
                else:
                    action = self.normal_actions(legal_actions)
                    return action

            if hand.has_three():
                if random.random() < self.bluff_threshold:
                    action = self.aggressive_actions(legal_actions)
                    return action
                else:
                    action = self.normal_actions(legal_actions)
                    return action

            if hand.has_two_pairs():
                if random.random() < self.bluff_threshold:
                    action = self.aggressive_actions(legal_actions)
                    return action
                else:
                    action = self.normal_actions(legal_actions)
                    return action

            if hand.has_pair():
                if random.random() < self.bluff_threshold:
                    action = self.aggressive_actions(legal_actions)
                    return action
                else:
                    action = self.normal_actions(legal_actions)
                    return action


            if hand.has_high_card():
                if self.env.game.round_counter == 1 or self.env.game.round_counter == 2:
                    if random.random() < self.bluff_threshold:
                        action = self.aggressive_actions(legal_actions)
                        return action
                    else:
                        action = self.normal_actions(legal_actions)
                        return action
                action = self.deffensive_actions(legal_actions)
                return action

            action = self.normal_actions(legal_actions)
        return action


    def aggressive_actions(self, legal_actions):
        if 'raise' in legal_actions:
            return 1
        elif 'call' in legal_actions:
            return 0
        elif 'check' in legal_actions:
            return 3
        else:  # fold
            return 2

    def normal_actions(self, legal_actions):
        if 'check' in legal_actions:
            return 3
        elif 'call' in legal_actions:
            return 0
        elif 'raise' in legal_actions:
            return 1
        else:  # raise only on pairs and three
            return 2

    def deffensive_actions(self, legal_actions):
        if 'fold' in legal_actions:
            return 2
        elif 'check' in legal_actions:
            return 3
        elif 'call' in legal_actions:
            return 0
        else:  # raise only on pairs and three
            return 1

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the threshold agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        # probs = [0 for _ in range(self.num_actions)]
        # for i in state['legal_actions']:
        #     probs[i] = 1/len(state['legal_actions'])

        info = {}
        #info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info