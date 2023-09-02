import collections
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from random import sample

from rlcard.utils.utils import remove_illegal


@dataclass
class Trans:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class MYDQNAgent:
    '''
    Approximate clone of rlcard.agents.dqn_agent.DQNAgent
    that depends on PyTorch instead of Tensorflow
    '''
    def __init__(self,
                 model,
                 env,
                 model_path='./Dqn_model',
                 epsilon_decay=0.995,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 card_obs_shape=(6, 13, 4),
                 action_obs_shape=(24, 4, 3),
                 learning_rate=0.00005,
                 num_actions=4,
                 device=None):

        self.num_actions = num_actions
        self.env = env
        self.model_path = model_path
        self.eps_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.card_obs_shape = card_obs_shape
        self.action_obs_shape = action_obs_shape
        self.alpha = learning_rate

        self.model = Model(card_obs_shape, action_obs_shape, num_actions, self.alpha)
        self.tgt = Model(card_obs_shape, action_obs_shape, num_actions, self.alpha)

        self.rb = ReplayBuffer

        self.qualities = collections.defaultdict(list)


        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


    def train(self):
        ''' Do one iteration of QLA
        '''
        self.iteration += 1
        self.env.reset()
        self.find_agent()
        v = self.traverse_tree()
        self._decay_epsilon()

    def _decay_epsilon(self):
        ''' Decay epsilon
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)

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
            return None, chips[self.agent_id], True

        current_player = self.env.get_player_id()
        # other agent move
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            action = self.env.agents[current_player].step(state)

            # Keep traversing the child state
            self.env.step(action)
            next_state, reward, done = self.traverse_tree()
            self.env.step_back()
            return next_state, reward, done

        if current_player == self.agent_id:
            card_obs, state_obs, legal_actions = self.get_state(current_player)
            rewards = np.zeros(len(legal_actions))
            cur_state = tuple(card_obs, state_obs)
            i = 0

            # if first time we encounter state initialize qualities or get the previous policy
            self.action_probs(card_obs, state_obs, self.policy, self.qualities)

            for i, action in enumerate(legal_actions, start=0):
                # Keep traversing the child state
                self.env.step(action)
                next_state, reward, done = self.traverse_tree()
                self.env.step_back()

                trans = Trans(cur_state, action, reward, next_state, done)
                self.rb.insert(trans)

                rewards[i] = reward

        return cur_state, rewards.mean(), False





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


    """Pure NN"""

    def update_tgt_model(self, model, tgt):
        tgt.load_state_dict(model.state_dict())

    def get_actions(self, card_obs, action_obs):
        # obs shape is (N, card_tensor, state_tensor)
        # q_vals = (N, 4)
        q_vals = self.model(card_obs, action_obs)

        return q_vals.max(-1)[1]

class Model(nn.Module):
    """Our network"""
    def __init__(self, card_obs_shape, action_obs_shape, num_actions, learning_rate=0.0005):
        super(Model, self).__init__()
        self.lr = learning_rate
        self.card_obs_shape = card_obs_shape
        self.action_obs_shape = action_obs_shape
        self.num_actions = num_actions
        input_dim1 = card_obs_shape[0] * card_obs_shape[1] * card_obs_shape[2]
        input_dim2 = action_obs_shape[0] * action_obs_shape[1] * action_obs_shape[2]

        self.layer_path1 = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim1, 512),
            nn.ReLU(),
        )

        self.layer_path2 = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim2, 512),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            nn.Linear(2 * 512, num_actions),
        )

        self.opt = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, obs1, obs2):
        # Process each observation through its respective pathway
        obs1_out = self.layer_path1(obs1)
        obs2_out = self.layer_path2(obs2)

        # Concatenate the outputs from both pathways
        combined_features = torch.cat((obs1_out, obs2_out), dim=1)

        # Pass the combined features through the final layer
        actions = self.final_layer(combined_features)
        return actions

    def train_step(self, state_transitions, model, tgt, num_actions):
        cur_states = torch.stack(torch.Tensor(s.state) for s in state_transitions)
        rewards = torch.stack(torch.Tensor(s.reward) for s in state_transitions)
        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])
        next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions])
        actions = [s.action for s in state_transitions]

        with torch.no_grad():
            qvals_next = tgt(next_states).max(-1)[0]  # (N, num_actions)

        model.opt.zero_grad()
        qvals = model(cur_states)  # (N, num_actions)
        one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)
        loss = (rewards + mask[:, 0]*qvals_next - torch.sum(qvals * one_hot_actions, dim=-1)).mean
        loss.backward()
        model.opt.step()
        return loss




class ReplayBuffer:
    """ Our replay buffer
    """
    def __init__(self, buffer_size=1000000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def insert(self, trans):
        self.buffer.append(trans)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)






















