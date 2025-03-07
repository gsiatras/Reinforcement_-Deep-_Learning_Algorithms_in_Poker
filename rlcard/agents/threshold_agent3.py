import numpy as np
from rlcard.games.newlimitholdem2.utils import Hand


class ThresholdAgent3(object):
    ''' A threshold agent. He will be playing new-limit-holdem.
        Will bet the maximum amount allowed in each round, provided that it has at least a high enough "combination".
        In round 1 it will always bet/raise with a Q, K or Ace.
        In round 2 it will always bet/raise with any pair and three.
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        # return np.random.choice(list(state['legal_actions'].keys()))
        legal_actions = state['raw_legal_actions'];

        if len(state['raw_obs']['public_cards']) == 0:  #we are on 1st round i.e. no public cards
            hand = state['raw_obs']['hand'][0]
            if hand == 'SA' or hand == 'HA' or hand == 'DA' or hand == 'CA' or\
                hand == 'SK' or hand == 'HK' or hand == 'DK' or hand == 'CK' or \
                hand == 'SQ' or hand == 'HQ' or hand == 'DQ' or hand == 'CQ':
                #high enough combination
                #i will play as offensive as I can i.e. 1raise 2. call 3. check
                if 'raise' in legal_actions:
                    return 1
                elif 'call' in legal_actions:
                    return 0
                elif 'check' in legal_actions:
                    return 3
                else:   #fold
                    return 2
            else:   # play randomly
                if 'check' in legal_actions:
                    return 3
                elif 'call' in legal_actions:
                    return 0
                elif 'raise' in legal_actions:
                    return 1
                else:  # fold
                    return 2
        else:   # second round raise only on pairs and threes
            cards = state['raw_obs']['hand'] + state['raw_obs']['public_cards']
            hand = Hand(cards)
            hand.evaluateHand()
            if hand.has_three() or hand.has_pair():
                if 'raise' in legal_actions:
                    return 1
                elif 'call' in legal_actions:
                    return 0
                elif 'check' in legal_actions:
                    return 3
                else:   #fold
                    return 2
            else:
                if 'check' in legal_actions:
                    return 3
                elif 'call' in legal_actions:
                    return 0
                elif 'fold' in legal_actions:
                    return 2
                else:  # raise only on pairs and three
                    return 1

    def get_action_probs(self, state, num_actions):
        '''Get the action probs of the agent
        '''

        legal_actions = state['raw_legal_actions']
        legal_actions1 = list(state['legal_actions'].keys())
        action_probs = np.zeros(num_actions)
        if len(state['raw_obs']['public_cards']) == 0:  # we are on 1st round i.e. no public cards
            hand = state['raw_obs']['hand'][0]
            if hand == 'SA' or hand == 'HA' or hand == 'DA' or hand == 'CA' or \
                    hand == 'SK' or hand == 'HK' or hand == 'DK' or hand == 'CK':
                if 'raise' in legal_actions:
                    action_probs[1] = 1
                    return action_probs
                elif 'call' in legal_actions:
                    action_probs[0] = 1
                    return action_probs
                elif 'check' in legal_actions:
                    action_probs[3] = 1
                    return action_probs
                else:  # fold
                    action_probs[2] = 1
                    return action_probs
            else:  # play randomly
                i = len(legal_actions)
                for a in legal_actions1:
                    action_probs[a] = 1 / i
                return action_probs
        else:
            cards = state['raw_obs']['hand'] + state['raw_obs']['public_cards']
            hand = Hand(cards)
            hand.evaluateHand()
            if hand.has_three() or hand.has_pair():
                if 'raise' in legal_actions:
                    action_probs[1] = 1
                    return action_probs
                elif 'call' in legal_actions:
                    action_probs[0] = 1
                    return action_probs
                elif 'check' in legal_actions:
                    action_probs[3] = 1
                    return action_probs
                else:  # fold
                    action_probs[2] = 1
                    return action_probs
            else:
                if 'check' and 'fold' in legal_actions:
                    action_probs[3] = 0.5
                    action_probs[2] = 0.5
                    return action_probs
                elif 'fold' and 'call' in legal_actions:
                    action_probs[0] = 0.5
                    action_probs[2] = 0.5
                    return action_probs
                elif 'check' in legal_actions:
                    action_probs[3] = 1
                    return action_probs
                elif 'fold' in legal_actions:
                    action_probs[2] = 1
                    return action_probs
                elif 'call' in legal_actions:
                    action_probs[0] = 1
                    return action_probs
                else:  # raise only on pairs and three
                    action_probs[1] = 1
                    return action_probs

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