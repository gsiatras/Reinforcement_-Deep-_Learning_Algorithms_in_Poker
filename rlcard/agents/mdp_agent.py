import collections

from numpy import random
from scipy.special import softmax
import os
import pickle

from rlcard.utils.utils import *


class MDPAgent:
    ''' Implement policy - iteration algorithm
    '''

    def __init__(self, env, g=1):
        ''' Initialize Agent
dp
         Args:
         env (Env): Env class
         converge se 4 iterations
        '''

        self.gamma = g
        self.agent_id = 0
        self.use_raw = False
        self.env = env
        self.rank_list = ['A', 'T', 'J', 'Q', 'K']
        self.card_prob = 0.2

        # A policy is a dict state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.state_values = collections.defaultdict(list)
        self.iteration = 0
        self.flag = 0
        self.flag2 = 0
        self.rank = None
        self.public_ranks = None

    def train(self, episodes=None):
        ''' Find optimal policy
        '''
        while True:
            #k += 1
            self.iteration += 1
            print(self.iteration)
            old_policy = self.policy.copy()
            self.evaluating_phase()
            if self.compare_policys(old_policy, self.policy):
                break
            if self.iteration == 10:
                break
        print('Optimal policy found: State space length: %d after %d iterations' % (len(self.policy), self.iteration))


    def compare_policys(self, p1, p2):
        if p1.keys() != p2.keys():
            print('dif pol keys')
            return False
        count = 0
        for key in p1:
            if not np.array_equal(p1[key], p2[key]):
                count += 1
        if count > 0:
            print('changes in policy: %d' % count)
            return False
        return True

    def compare_values(self, v1, v2):
        if v1.keys() != v2.keys():
            print('dif value keys')
            return False
        count = 0
        for key in v1:
            if v1[key] != v2[key]:
                count += 1
        if count > 0:
            print('changes in values: %d' % count)
            return False
        return True

    def evaluating_phase(self):
        self.evaluate_policy(True)
        # while True:
        #     old_values = self.state_values
        #     self.evaluate_policy()
        #     if self._compare_values(old_values, self.state_values):
        #         break

    def first_run(self):
        if self.env.is_over():
            return

        self.roundzero()
        current_player = self.env.get_player_id()
        # compute the q of previous state
        if not current_player == self.agent_id:
            obs, legal_actions = self.get_state(current_player)
            for action in legal_actions:
                # Keep traversing the child state
                self.env.step(action)
                self.first_run()
                self.env.step_back()
            return

        if current_player == self.agent_id:
            Vstate = 0
            obs, legal_actions = self.get_state(current_player)
            # initialize a random policy for state
            action_probs = self.action_probs(obs, legal_actions, self.policy)
            for action in legal_actions:
                # Keep traversing the child state
                self.env.step(action)
                self.first_run()
                self.env.step_back()
            self.state_values[obs] = Vstate
        return

    def evaluate_policy(self, first=None):
        '''Run through the game to initialize the state space, the random policy, and the Value function
        of random policy'''

        if first is None:
            self.find_agent()
            suit = 'S'
            Vtotal = 0
            for rank1 in self.rank_list:
                for rank2 in self.rank_list:
                    for rank3 in self.rank_list:
                        self.env.reset(self.agent_id, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3))
                        Vtotal += self.evaluate_tree()
            player = (self.agent_id + 1) % self.env.num_players
            for rank1 in self.rank_list:
                for rank2 in self.rank_list:
                    for rank3 in self.rank_list:
                        self.env.reset(player, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3))
                        Vtotal += self.evaluate_tree()
            print(Vtotal)
            return Vtotal
        else:
            self.find_agent()
            suit = 'S'
            for rank1 in self.rank_list:
                for rank2 in self.rank_list:
                    for rank3 in self.rank_list:
                        self.env.reset(self.agent_id, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3))
                        self.first_run()
            player = (self.agent_id + 1) % self.env.num_players
            for rank1 in self.rank_list:
                for rank2 in self.rank_list:
                    for rank3 in self.rank_list:
                        self.env.reset(player, self.agent_id, Card(suit, rank1), Card(suit, rank2),
                                       Card(suit, rank3))
                        self.first_run()

    def evaluate_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            return chips[self.agent_id]

        current_player = self.env.get_player_id()
        # compute the q of previous state
        if not current_player == self.agent_id:
            vtotal = 0
            if self.env.op_has_card(current_player):
                self.flag = 1
                for rank in self.rank_list:
                    self.rank = rank
                    self.env.change_op_hand(Card('S', rank), current_player)
                    # other agent move
                    obs, legal_actions = self.get_state(current_player)
                    state = self.env.get_state(current_player)
                    action_probs = self.env.agents[current_player].get_action_probs(state, self.env.num_actions)
                    Vstate = 0
                    for action in legal_actions:
                        prob = action_probs[action]
                        # Keep traversing the child state
                        self.env.step(action)
                        v = self.evaluate_tree()
                        Vstate += v * prob
                        self.env.step_back()
                    vtotal += Vstate*self.card_prob
                self.flag = 0
                return vtotal*self.gamma
            else:
                # other agent move
                obs, legal_actions = self.get_state(current_player)
                state = self.env.get_state(current_player)
                action_probs = self.env.agents[current_player].get_action_probs(state, self.env.num_actions)
                Vstate = 0
                for action in legal_actions:
                    prob = action_probs[action]
                    # Keep traversing the child state
                    self.env.step(action)
                    v = self.evaluate_tree()
                    Vstate += v * prob
                    self.env.step_back()
                return Vstate*self.gamma

        if current_player == self.agent_id:
            quality = {}
            new_Vstate = 0
            Vstate = 0
            obs, legal_actions = self.get_state(current_player)
            # if first time we encounter state initialize qualities
            action_probs = self.action_probs(obs, legal_actions, self.policy)

            for action in legal_actions:
                prob = action_probs[action]
                # Keep traversing the child state
                self.env.step(action)
                v = self.evaluate_tree()
                self.env.step_back()

                quality[action] = v  # Qvalue
                Vstate += v*prob

            self.state_values[obs] = Vstate


            ''' alter policy by choosing the action with the max value'''
            # self.improve_policy(obs, quality, legal_actions)
            # new_action_probs = self.action_probs(obs, legal_actions, self.policy)
            # for action in legal_actions:
            #     prob = new_action_probs[action]
            #     new_Vstate += prob*quality[action]

        return new_Vstate * self.gamma


    def action_probs(self, obs, legal_actions, policy):
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

        if self.flag == 1:
            obs1 = (obs, self.rank)
            # if new state initialize policy
            if obs not in policy.keys() and obs1 not in policy.keys():
                best_action = random.choice(legal_actions)
                #best_action = np.argmax(tactions)
                action_probs = np.array([0 for action in range(self.env.num_actions)])
                action_probs[best_action] = 1
                self.policy[obs1] = action_probs
            elif obs1 not in policy.keys():
                action_probs = policy[obs].copy()
                print('1000')
            else:
                action_probs = policy[obs1].copy()
                print('1000')
        else:
            if obs not in policy.keys():
                best_action = random.choice(legal_actions)
                #best_action = np.argmax(tactions)
                action_probs = np.array([0 for action in range(self.env.num_actions)])
                action_probs[best_action] = 1
                self.policy[obs] = action_probs
            else:
                action_probs = policy[obs].copy()
                print('1000')

        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs



    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
            state (numpy.array): State representation

        Returns:
            action (int): Predicted action
            info (dict): A dictionary containing information
        '''

        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.policy)
        # action = np.random.choice(len(probs), p=probs)
        action = np.argmax(probs)

        # if np.random.rand() < self.epsilon:
        #     action = np.random.choice(list(state['legal_actions'].keys()))
        # else:
        #     action = np.argmax(probs)
        #
        # self.decay_epsilon()

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

    def find_agent(self):
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, MDPAgent):
                self.agent_id = id
                break


    def roundzero(self):
        if self.env.first_round():
            self.flag2 = 1
        else:
            self.flag2 = 0