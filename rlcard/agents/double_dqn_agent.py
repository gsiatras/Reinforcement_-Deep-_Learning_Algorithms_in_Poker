import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from dataclasses import dataclass
from typing import Any
from random import sample
from rlcard.utils.utils import *
import pickle
import torch.nn.init as init


@dataclass
class Trans:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class DoubleDQNAgent(object):
    '''
    DOUBLE DQN AGENT for limit texas holdem
    '''
    def __init__(self,
                 env,
                 model_path='./Dqn_model',
                 epsilon_decay=0.9999,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 card_obs_shape=(6, 4, 13),
                 action_obs_shape=(24, 3, 4),
                 learning_rate=0.00025,
                 num_actions=4,
                 batch_size=128,
                 tgt_update_freq=10000,
                 train_steps=1,
                 buffer_size=100000,
                 device=None,
                 self_play=False):

        self.num_actions = num_actions
        self.env = env
        self.model_path = model_path
        self.eps_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.card_obs_shape = card_obs_shape
        self.action_obs_shape = action_obs_shape
        self.learning_rate = learning_rate
        self.tgt_update_freq = tgt_update_freq
        self.num_train_steps = train_steps
        self.batch_size = batch_size
        self.agent_id = 0
        self.use_raw = False
        self.buffer_size = buffer_size

        self.model = Model(card_obs_shape, action_obs_shape, num_actions, self.learning_rate)  # model
        self.model.initialize_weights()

        self.tgt = Model(card_obs_shape, action_obs_shape, num_actions, self.learning_rate)  # target model
        self.tgt.initialize_weights()

        self.rb = ReplayBuffer(self.buffer_size)
        self.episodes = 0
        self.losses = []  # Store losses for monitoring
        self.start_pos = []
        self.self_play = self_play

        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        if torch.cuda.is_available():
            # Get the CUDA version
            cuda_version = torch.version.cuda
            print(f"CUDA Version: {cuda_version}")
        else:
            print("CUDA is not available on this system.")


    def train(self):
        ''' Do one iteration of QLA
        '''
        self.episodes += 1
        self.env.reset()
        self.find_agent()
        if not self.self_play:
            self.traverse_tree()
        else:
            self.train_self_play()
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
            if isinstance(agent, DoubleDQNAgent):
                self.agent_id = id
                break
        self.start_pos.append(self.agent_id)

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

    def train_self_play(self):
        '''Self play training loop
        '''
        # Check if the game is over to return the chips earned(reward of the game)
        current_player = self.env.get_player_id()
        if self.env.is_over():
            card_obs, action_obs, legal_actions = self.get_state(current_player)
            state = (card_obs, action_obs)
            chips = self.env.get_payoffs()
            return state, state, chips, True

        # make move
        card_obs, action_obs, legal_actions = self.get_state(current_player)
        cur_state = (card_obs, action_obs)

        obs1, obs2 = self.prepare_data(card_obs, action_obs)
        with torch.no_grad():
            qvals = self.model(obs1, obs2)
            qvals = self.remove_illegal(qvals[0], legal_actions)

        if np.random.rand() < self.epsilon:
            # explore
            action = np.random.choice(legal_actions)
        else:
            # action with highest Q value
            action = np.argmax(qvals)

        # print(action, legal_actions)

        self.env.step2(action, legal_actions)
        my_next_state, other_next_state, rewards, done = self.train_self_play()
        self.env.step_back()
        agent_rewards = rewards[current_player]
        rewards[current_player] = 0
        trans = Trans(cur_state, action, agent_rewards, my_next_state, done)
        self.rb.insert(trans)

        return other_next_state, cur_state, rewards, False

    def traverse_tree(self):
        ''' Traverse the game tree:

        Check if the game is over to return the chips earned(reward of the game)
        If opponents plays make the other agent play
        If our agent plays get the Q value of the state from model
        epsilon policy
        Store transitions to buffer
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
                qvals = self.remove_illegal(qvals[0], legal_actions)
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

        return cur_state, 0, False

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
        '''Prepare data to enter the net (flatten)
               Args:
                   card_state (card_state_shape = (x,y,z))
                   action_state (action_state_shape = (x,y,z))
               Returns:
                   card_state_flat (card_state_shape = (1,(1,x*y*z))
                   action_state_flat (action_state_shape = (1,(x*y*z))
        '''
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
        '''
        Update the weights of the target model
        '''
        tgt.load_state_dict(model.state_dict())


    def train_step(self, state_transitions, model, tgt, num_actions):
        '''
        Train the model
        Args:
            state_transitions: a batch of transitions
            model: our model
            tgt: our target model
            num_actions: num of actions, output of the net
        Returns:
            loss: loss of the update
            loss according to DQN paper
        '''

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
        '''
        Evaluating step
        Args:
            state: current state
        Returns:
            action (int)
        '''
        card_obs = state['card_tensor']
        action_obs = state['action_tensor']
        legal_actions = list(state['legal_actions'].keys())

        obs1, obs2 = self.prepare_data(card_obs, action_obs)
        with torch.no_grad():
            qvals = self.tgt(obs1, obs2)
            qvals = self.remove_illegal(qvals[0], legal_actions)

        action = np.argmax(qvals)

        info = {}
        info['qvals'] = {state['raw_legal_actions'][i]: float(qvals[list(state['legal_actions'].keys())[i]]) for i in
                         range(len(state['legal_actions']))}

        return action, info


    def step(self, state):
        '''step = eval.step
        '''
        return self.eval_step(state)


    def save(self, filename='checkpoint_my_dqn_dueling.pt'):
        """
        Save relevant parameters of the MYDQNAgent instance to a file.
        """
        agent_params = {
            'epsilon': self.epsilon,
            'model_state_dict': self.model.state_dict(),
            'tgt_state_dict': self.tgt.state_dict(),
            'rb_buffer': self.rb.buffer,
            'episodes': self.episodes,
            'losses': self.losses,
            "agent_id": self.agent_id,

            "num_actions": self.num_actions,
            "env": self.env,
            "model_path": self.model_path,
            "epsilon_decay": self.eps_decay,
            "epsilon_min": self.epsilon_min,
            "card_obs_shape": self.card_obs_shape,
            "action_obs_shape": self.action_obs_shape,
            "learning_rate": self.learning_rate,
            "tgt_update_freq": self.tgt_update_freq,
            "num_train_steps": self.num_train_steps,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "use_raw": self.use_raw,
            "device": self.device
        }

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        checkpoint_file = open(os.path.join(self.model_path, filename), 'wb')
        pickle.dump(agent_params, checkpoint_file)
        checkpoint_file.close()

    @classmethod
    def load(cls, model_path, filename='checkpoint_my_dqn_dueling.pt'):
        ''' Load a model
        Returns: instance if found
                    else
                none
        '''
        try:
            file = open(os.path.join(model_path, filename), 'rb')
            agent_params = pickle.load(file)
            file.close()
            print("\nINFO - Restoring model from checkpoint...")

            agent_instance = cls(
                env=agent_params["env"],
                model_path=agent_params["model_path"],
                epsilon_decay=agent_params["epsilon_decay"],
                epsilon_start=agent_params["epsilon"],
                epsilon_end=agent_params["epsilon_min"],
                card_obs_shape=agent_params["card_obs_shape"],
                action_obs_shape=agent_params["action_obs_shape"],
                learning_rate=agent_params["learning_rate"],
                num_actions=agent_params["num_actions"],
                batch_size=agent_params["batch_size"],
                tgt_update_freq=agent_params["tgt_update_freq"],
                train_steps=agent_params["num_train_steps"],
                buffer_size=agent_params["buffer_size"],
                device=agent_params["device"]
            )

            agent_instance.losses = agent_params["losses"]
            agent_instance.agent_id = agent_params["agent_id"]
            agent_instance.episodes = agent_params["episodes"]
            agent_instance.tgt.load_state_dict(agent_params['tgt_state_dict'])
            agent_instance.model.load_state_dict(agent_params['model_state_dict'])
            agent_instance.rb.buffer = agent_params['rb_buffer']

            return agent_instance
        except FileNotFoundError:
            print(f"\nINFO - No checkpoint file '{filename}' found. Creating a new agent.")
            return None


class Model(nn.Module):
    """Our network
    2 layer paths (action space and card space 2 layers each 512neurons)
    1 final layer merging the 2 others (2 layers 512 neurons)
    """
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
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.layer_path2 = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.final_layer = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

        self.opt = optim.Adam(self.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, obs1, obs2):
        '''Forward propagation
        '''
        #print(obs1.shape)

        # Process each observation through its respective pathway
        obs1_out = self.layer_path1(obs1)
        obs2_out = self.layer_path2(obs2)

        # Concatenate the outputs from both pathways
        combined_features = torch.cat((obs1_out, obs2_out), dim=1)

        # Pass the combined features through the final layer
        actions = self.final_layer(combined_features)
        return actions
    def initialize_weights(self):
        '''
        Initialize net weights
        random small value for W
        0 for bias
        '''
        for layer in self.layer_path1:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight, gain=1)
                # init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

        for layer in self.layer_path2:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight, gain=1)
                # init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)

        for layer in self.final_layer:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight, gain=1)
                # init.normal_(layer.weight, mean=0, std=0.01)
                init.constant_(layer.bias, 0)



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


























