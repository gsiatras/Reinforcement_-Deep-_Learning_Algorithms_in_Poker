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
from rlcard.utils.utils import *

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
                 env,
                 model_path='./Dqn_model',
                 epsilon_decay=0.9999,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 card_obs_shape=(6, 4, 13),
                 action_obs_shape=(24, 3, 4),
                 learning_rate=0.005,
                 num_actions=4,
                 batch_size=64,
                 tgt_update_freq=700,
                 train_steps=1,
                 device=None):

        self.num_actions = num_actions
        self.env = env
        self.model_path = model_path
        self.eps_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.card_obs_shape = card_obs_shape
        self.action_obs_shape = action_obs_shape
        self.alpha = learning_rate
        self.tgt_update_freq = tgt_update_freq
        self.num_train_steps = train_steps
        self.batch_size = batch_size
        self.agent_id = 0
        self.use_raw = False

        self.model = Model(card_obs_shape, action_obs_shape, num_actions, self.alpha)
        self.tgt = Model(card_obs_shape, action_obs_shape, num_actions, self.alpha)

        self.rb = ReplayBuffer()
        self.episodes = 0
        self.losses = []  # Store losses for monitoring

        self.qualities = collections.defaultdict(list)

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device


    def train(self):
        ''' Do one iteration of QLA
        '''
        np.random.seed(2)
        self.episodes += 1
        self.env.reset()
        self.find_agent()
        self.traverse_tree()
        self._decay_epsilon()

        if self.rb.size() > self.batch_size:

            for _ in range(self.num_train_steps):
                loss = self.train_step(self.rb.sample(self.batch_size), self.model, self.tgt, self.num_actions)
                self.losses.append(loss.item())  # Convert the loss to a scalar and store it

            if self.episodes % self.tgt_update_freq == 0:
                self.update_tgt_model(self.model, self.tgt)

            # Print or log the average loss
            return loss.item()

    def find_agent(self):
        ''' Find if the agent starts first or second
        '''
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, MYDQNAgent):
                self.agent_id = id
                break

    def get_avg_loss(self):
        if len(self.losses) > 0:
            avg_loss = sum(self.losses) / len(self.losses)  # Calculate the average loss
        else:
            avg_loss = 0
        return avg_loss, self.epsilon

    def _decay_epsilon(self):
        ''' Decay epsilon
        '''
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.eps_decay)

    def traverse_tree(self):
        ''' Traverse the game tree:

        Check if the game is over to return the chips earned(reward of the game)
        If opponents plays make the other agent play
        If our agent plays check every possible action and get the Q value of the action
        Then return the Qvalue of the best or a random state according to the epsilon (off policy)
        Change the policy according to the new Q values
        '''
        current_player = self.env.get_player_id()
        # Check if the game is over to return the chips earned(reward of the game)
        if self.env.is_over():
            card_obs, action_obs, legal_actions = self.get_state(current_player)
            state = (card_obs, action_obs)
            chips = self.env.get_payoffs()
            return state, chips[self.agent_id], True

        # other agent move
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            action = self.env.agents[current_player].step(state)

            # Keep traversing the child state
            legal_actions = list(state['legal_actions'].keys())
            self.env.step2(action, legal_actions)
            next_state, reward, done = self.traverse_tree()
            self.env.step_back()
            return next_state, reward, done

        if current_player == self.agent_id:
            card_obs, action_obs, legal_actions = self.get_state(current_player)
            cur_state = (card_obs, action_obs)

            obs1, obs2 = self.prepare_data(card_obs, action_obs)
            with torch.no_grad():
                qvals = self.model(obs1, obs2)
                #print(qvals)
                #print(legal_actions)
                qvals = self.remove_illegal(qvals, legal_actions)
                #print(qvals)

            if np.random.rand() < self.epsilon:
                # explore
                action = np.random.choice(legal_actions)
                #print(action)
            else:
                # action with highest Q value
                action = np.argmax(qvals)

            #print(action, legal_actions)

            self.env.step2(action, legal_actions)
            next_state, reward, done = self.traverse_tree()
            self.env.step_back()

            trans = Trans(cur_state, action, reward, next_state, done)
            self.rb.insert(trans)

        return cur_state, reward, False

    def remove_illegal(self, qvals, legal_actions):
        """turn back to np array and remove illegal actions
        """

        qvals = qvals.numpy()
        modified_qvals = qvals.copy()

        # Set the Q-values of illegal actions to negative infinity
        illegal_actions = [i for i in range(len(qvals)) if i not in legal_actions]
        modified_qvals[illegal_actions] = -np.inf

        return modified_qvals

    def prepare_data(self, obs1, obs2):
        # Convert to float if not already
        obs1_tensor = torch.from_numpy(obs1).float()
        obs2_tensor = torch.from_numpy(obs2).float()
        # Add a batch dimension of size 1
        obs1_tensor = obs1_tensor.unsqueeze(0)
        obs2_tensor = obs2_tensor.unsqueeze(0)

        # Flatten the tensor into a 1D vector
        obs1_flattened = obs1_tensor.view(1, -1)
        obs2_flattened = obs2_tensor.view(1, -1)

        #print(obs1_flattened.shape)

        return obs1_flattened, obs2_flattened


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
        return state['card_tensor'], state['action_tensor'], list(state['legal_actions'].keys())

    """Pure NN"""

    def update_tgt_model(self, model, tgt):
        tgt.load_state_dict(model.state_dict())

    def get_actions(self, card_obs, action_obs):
        # obs shape is (N, card_tensor, state_tensor)
        # q_vals = (N, 4)
        q_vals = self.model(card_obs, action_obs)

        return q_vals.max(-1)[1]

    def train_step(self, state_transitions, model, tgt, num_actions):
        #print(state_transitions)
        # Unpack the state transitions
        cur_state_card_trans = [s.state[0] for s in state_transitions]
        cur_state_action_trans = [s.state[1] for s in state_transitions]

        next_state_card_trans = [s.next_state[0] for s in state_transitions]
        next_state_action_trans = [s.next_state[1] for s in state_transitions]

        cur_states1 = torch.stack([torch.Tensor(state) for state in cur_state_card_trans])
        cur_states2 = torch.stack([torch.Tensor(state) for state in cur_state_action_trans])
        rewards = [s.reward for s in state_transitions]
        #print(rewards)
        rewards = torch.tensor(rewards, dtype=torch.int32)
        mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])
        next_states1 = torch.stack([torch.Tensor(state) for state in next_state_card_trans])
        next_states2 = torch.stack([torch.Tensor(state) for state in next_state_action_trans])
        actions = [s.action for s in state_transitions]

        with torch.no_grad():
            qvals_next = tgt(next_states1, next_states2).max(-1)[0]  # (N, num_actions)

        model.opt.zero_grad()
        qvals = model(cur_states1, cur_states2)  # (N, num_actions)
        one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions)
        loss = (rewards + mask[:, 0]*qvals_next - torch.sum(qvals * one_hot_actions, dim=-1)) ** 2
        mean_loss = loss.mean()
        #loss = model.mse_loss(qvals, qvals_next)
        mean_loss.backward()
        model.opt.step()
        return mean_loss

    def eval_step(self, state):
        card_obs = state['card_tensor']
        action_obs = state['action_tensor']
        legal_actions = list(state['legal_actions'].keys())

        obs1, obs2 = self.prepare_data(card_obs, action_obs)
        with torch.no_grad():
            qvals = self.tgt(obs1, obs2)
            qvals = self.remove_illegal(qvals, legal_actions)

        action = np.argmax(qvals)

        info = {}
        info['qvals'] = {state['raw_legal_actions'][i]: float(qvals[list(state['legal_actions'].keys())[i]]) for i in
                         range(len(state['legal_actions']))}

        return action, info


    def step(self, state):
        '''step = eval.step
        '''
        return self.eval_step(state)

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
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, obs1, obs2):
        #print(obs1.shape)

        # Process each observation through its respective pathway
        obs1_out = self.layer_path1(obs1)
        obs2_out = self.layer_path2(obs2)

        # Concatenate the outputs from both pathways
        combined_features = torch.cat((obs1_out, obs2_out), dim=1)

        # Pass the combined features through the final layer
        actions = self.final_layer(combined_features)
        return actions[0]


class ReplayBuffer:
    """ Our replay buffer
    """
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def insert(self, trans):
        self.buffer.append(trans)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)

    def size(self):
        return len(self.buffer)
























