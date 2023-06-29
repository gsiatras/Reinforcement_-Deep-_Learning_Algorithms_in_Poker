import collections
from scipy.special import softmax
import os
import pickle

from rlcard.utils.utils import *


class QLAgent:
    ''' Implement Q-learning algorithm
    '''

    def __init__(self, env, model_path='./ql_model', a=0.1, g=1, e=0.99):
        ''' Initialize Agent
dp
         Args:
         env (Env): Env class
        '''
        self.epsilon = 1
        self.gamma = g
        self.alpha = a
        self.decay_factor = e
        self.agent_id = 0
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        self.epsilon_min = 0.01
        self.v = 0

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)

        # Regret is a dict state_str -> action regrets
        self.qualities = collections.defaultdict(list)

        self.iteration = 0

    def train(self):
        ''' Do one iteration of QLA
        '''
        self.iteration += 1
        self.env.reset()
        self.find_agent()
        v = self.traverse_tree()
        self._decay_epsilon()
        self.v = v

    def find_agent(self):
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, QLAgent):
                self.agent_id = id
                break

    def _decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.decay_factor)

    def traverse_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        current_player = self.env.get_player_id()
        # compute the quality of previous state
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
            obs, legal_actions = self.get_state(current_player)
            # if first time we encounter state initialize qualities
            self.action_probs(obs, legal_actions, self.policy, self.qualities)

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

    def action_probs(self, obs, legal_actions, policy, action_values):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy
            action_values (dict): The action_values of policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        # if new state initialize qualities and policy
        if obs not in policy.keys() and obs not in self.qualities.keys():
            tactions = np.array([-np.inf for action in range(self.env.num_actions)])
            for action in range(self.env.num_actions):
                if action in legal_actions:
                    tactions[action] = 0
            self.qualities[obs] = tactions
            action_probs = softmax(tactions)
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs].copy()
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs


    def update_policy(self, obs, next_state_values, legal_actions):
        ''' Update the policy according to the new state/action quality
                Args:
                    obs (str): state_str
                    next_state_values (list): The new qualities of the current iteration
         '''
        # update the quality function
        qf = self.qualities[obs]
        for i in next_state_values:
            qf[i] += self.alpha * (next_state_values[i] - qf[i])

        # update action values
        self.qualities[obs] = qf
        self.policy[obs] = softmax(qf)

    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''

        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.policy,
                                  self.qualities)
        # action = np.random.choice(len(probs), p=probs)
        action = np.argmax(probs)

        # if np.random.rand() < self.epsilon:
        #     action = np.random.choice(list(state['legal_actions'].keys()))
        # else:
        #     action = np.argmax(probs)
        #
        # self.decay_epsilon()
        self.find_agent()
        # other_agent = (self.agent_id + 1) % self.env.num_players
        # cardrank = self.env.get_card(other_agent)
        # print(cardrank)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in
                         range(len(state['legal_actions']))}

        return action, info

    def step(self, state):
        '''step = eval.step
        '''
        return self.eval_step(state)


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
        return state['obs'].tostring(), list(state['legal_actions'].keys())

    def save(self):
        '''  Save model
        '''

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        values_file = open(os.path.join(self.model_path, 'qualities.pkl'),'wb')
        pickle.dump(self.qualities, values_file)
        values_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''

        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'rb')
        self.policy = pickle.load(policy_file)
        policy_file.close()

        values_file = open(os.path.join(self.model_path, 'qualities.pkl'),'rb')
        self.qualities = pickle.load(values_file)
        values_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()