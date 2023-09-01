import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy

from rlcard.utils.utils import remove_illegal

class MYDQNAgent:
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 env,
                 model_path='./Dqn_model',
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 state_shape=None,
                 learning_rate=0.00005
                 ,):

        self.env = env
        self.model_path = model_path
        self.discount_factor = discount_factor
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.state_shape = state_shape
        self.alpha = learning_rate

        self.qualities = collections.defaultdict(list)


    def train(self):
        ''' Do one iteration of QLA
        '''
        self.iteration += 1
        self.env.reset()
        self.find_agent()
        v = self.traverse_tree()
        self._decay_epsilon()




    def traverse_tree(self):
        ''' Traverse the game tree:

        Check if the game is over to return the chips earned(reward of the game)
        If opponents plays make the other agent play
        If our agent plays check every possible action and get the Q value of the action
        Then return the Qvalue of the best or a random state according to the epsilon (off policy)
        Change the policy according to the new Q values
        '''

        # Check if the game is over to return the chips earned(reward of the game)
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        current_player = self.env.get_player_id()

        # other agent move
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            action = self.env.agents[current_player].step(state)

            # Keep traversing the child state
            self.env.step(action)
            Vstate = self.traverse_tree()
            self.env.step_back()
            return Vstate * self.gamma

        if current_player == self.agent_id:
            quality = {}
            card_obs, state_obs, legal_actions = self.get_state(current_player)
            # if first time we encounter state initialize qualities or get the previous policy
            self.action_probs(card_obs, state_obs, self.policy, self.qualities)

            for action in legal_actions:
                # Keep traversing the child state
                self.env.step(action)
                q = self.traverse_tree()
                self.env.step_back()

                quality[action] = q  # value of each action

            ''' alter policy according to new Vactions'''
            if np.random.rand() < self.epsilon:
                # explore
                qstate = np.random.choice(list(quality.values()))
            else:
                # action with highest Q value
                qstate = np.max(list(quality.values()))

            ''' alter Qfunction according to Q_next_state'''
            self.update_policy(obs, quality, legal_actions)

        return qstate * self.gamma


    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
            player_id (int): The player id

        Returns:
            (tuple) that contains:
                state (str): The state str
                legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['card_tensor'].tostring(), state['action_tensor'].tostring(), list(state['legal_actions'].keys())
