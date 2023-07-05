import pprint as pp
import numpy as np
import collections
from scipy.special import softmax
import os
import pickle

from rlcard.utils.utils import *
'''
P is the state space that we need for implementing value iteration
P["state1","raise"] for example captures what happens if at state1 I take action: raise.
next state is about the next state the agent will be. It's not about the other player's state


P = {

"state1": {
    "raise": [ { "state2": [0.9, 1.0, 9],           # next state: [prob of next state, reward of next state, num of times visited next state right after state1] 
                 "state3": [0.1, 0.5, 1]
               }, ctr                           ctr: (num of times action raise was taken when in state1)
    ],
    "check": [ { "state2": [0.1, 1.0, 1],
                 "state3": [0.8, 0.5, 8],
                 "state4": [0.1, 0.0, 1]
               }, ctr
    ]
},

"state2": {
    "call": [ { "state6": [0.5, 0.0, 5],           # next state: [prob of next state, reward of next state, num of times visited next state right after state1] 
                 "state3": [0.5, 0.5, 5]
               }, ctr                           ctr: (num of times action call was taken when in state2)
    ],
    "fold": [ { "state5": [0.1, 1.0, 1],
                 "state3": [0.8, 0.5, 8],
                 "state4": [0.1, 0.0, 1]
               }, ctr
    ]
}

}

Q table captures the values of all actions for all possible states in the enviroment
In each game round we can't examine each possible state,
hence we reset to zero in each round only the values of action of states that we will take. The rest stay the same  
Q = {
    "state1": [2,3,4,2],    # "state1": [reward for action 0,reward for action 1, reward for action 2, reward for action 3]
    "state2": [2,5,2,6],    # "state2": [reward for action 0,reward for action 1, reward for action 2, reward for action 3]
    "state3": [2,3,4,2],    # "state3": [reward for action 0,reward for action 1, reward for action 2, reward for action 3]
}

V table captures the expected return of the best action for each state and also what action is provides that
available actions: {0,1,2,3} ->{call, raise, fold, check}
V = {
    "state1": [2.4, 0],     # "state1": [reward for best action when in state1, action to take when in state1]
    "state2": [-1,2],       # "state2": [reward for best action when in state2, action to take when in state2]
    "state3": [4, 3],       # "state3": [reward for best action when in state3, action to take when in state3]
}

'''

class ValueIterAgent:
    ''' An agent that will play according to value iteration algorithm,
        in order to find optimal policy
    '''

    def __init__(self, env, model_path='./vi_model', gamma=0.6, conv_limit=1e-10):
        ''' Initilize the value iteration agent

        Args:
            env (Env): Env class
        '''
        self.random_choices = 0
        self.value_choices = 0
        self.use_raw = 2
        self.env = env
        self.conv_limit = conv_limit
        self.gamma = gamma
        self.agent_id = 0
        self.model_path = model_path
        self.P = collections.defaultdict(dict)              # state space
        self.V = collections.defaultdict(float)    # value function for each state (expected return of the best action for each state)
        self.Q = collections.defaultdict(list)     # Q table
    

    def value_iteration_algo(self):
        iteration = 0
        for state in self.P:
                self.V[state] = [0,0]
        while True:
            for state in self.P:
                self.Q[state] = [0,0,0,0]
            for state in self.P:
                for action in self.P[state]:
                    for item in self.P[state][action][0].items(): # for every next state after current state obs taking certain action 
                        prob_next_st, rew_next_st, ctr = item[1]    
                        nxt_st = item[0]
                        self.Q[state][action] += prob_next_st * (rew_next_st + self.gamma * self.V[nxt_st][0])
            ll = list(self.Q.values())  # list of lists with Q values of each action per state
            q_vals = np.max(ll, axis = 1)    #maximum expected reward for each state as calculated in Q table
            v_vals = [item[0] for item in list(self.V.values())]    # list of rewards in V 
            if np.max(np.abs(np.subtract(q_vals,v_vals))) < self.conv_limit:
                # converged
                print('\nState space has {} different states'.format(len(self.V)))
                print('Value iteration converged after {} iterations'.format(iteration))
                break   # found convergence must stop
            # Since i have not converged, i set new V(s)
            q_vals_ind = np.argmax(ll, axis = 1)    # index of action which provides the maximum expected reward for each state as calculated in Q table
            for i, st in enumerate(self.Q):         # Setting V value for each state
                self.V[st][1] = q_vals_ind[i]       # action that provides maximum expected reward when at state st
                self.V[st][0] = q_vals[i]           # maximum expected reward when at state st
            iteration += 1


    def learn_env(self):
        ''' Play games to learn the enviroment
        '''
        self.env.reset()
        self.find_agent()
        self.traverse_tree()

    def find_agent(self):
        agents = self.env.get_agents()
        for id, agent in enumerate(agents):
            if isinstance(agent, ValueIterAgent):
                self.agent_id = id
                break

    def traverse_tree(self):
        if self.env.is_over():
            chips = self.env.get_payoffs()
            current_player = self.env.get_player_id()
            if current_player == self.agent_id:
                obs, legal_actions = self.get_state(current_player)
                return chips[self.agent_id], obs, True
            else:
                return chips[self.agent_id], "other player", True

        current_player = self.env.get_player_id()
        # compute the quality of previous state
        if not current_player == self.agent_id:
            state = self.env.get_state(current_player)
            # other agent move
            action = self.env.agents[current_player].step(state)

            # Keep traversing the child state
            self.env.step(action)
            Vstate, next_state, terminal = self.traverse_tree()
            self.env.step_back()
            return Vstate, next_state, terminal
        
        if current_player == self.agent_id:
            obs, legal_actions = self.get_state(current_player)
            # update state space
            self.update_P(obs, legal_actions)
            for action in legal_actions:
                # Keep traversing the child state
                self.env.step(action)
                q, next_state, terminal = self.traverse_tree()    # I want my next state, not opponent's state
                if terminal:                    # If next state is terminal we should update P 
                    if next_state == "other player":
                        # this is my last state, game is finished and i took last action e.g. fold
                        next_state, next_st_legal_actions = self.get_state(current_player)  # this way we pass in next state the info about action taken
                        self.update_P(next_state, next_st_legal_actions)    # to record the last state into dicts
                    else:
                        self.update_P(next_state, legal_actions)    # to record the last state into dicts
                    
                    # iterate in P to set reward of next state
                    # a state must give same reward in value iteration whenever it shows up
                    # for state in self.P:
                    #     for act in state:
                    for state in self.P:
                        for act in self.P[state]:
                            if next_state in self.P[state][act][0].keys():
                                self.P[state][act][0][next_state][1] = (self.P[state][act][0][next_state][1] + q) / 2     # q_new = (q_old + q) / 2

                self.env.step_back()
                
                self.P[obs][action][1] +=1  #took action when in state obs one more time
                if next_state not in self.P[obs][action][0].keys() and terminal:   #next state first time recorded for current state and it is terminal
                    self.P[obs][action][0][next_state] = [0, q, 1] #prob of next state, reward for this state, times visited this state 
                elif next_state not in self.P[obs][action][0].keys() and not terminal:
                    self.P[obs][action][0][next_state] = [0, 0, 1]
                else:   # I have visited again next state, after current state obs
                    self.P[obs][action][0][next_state][2] += 1 
                
                for i in self.P[obs][action][0]:    #calculate again probabilities of each recorded next state when in current state obs and taken certain action
                    self.P[obs][action][0][i][0] = self.P[obs][action][0][i][2] / self.P[obs][action][1]    #times visited next state/sum of all visits

            return q, obs, False

    @staticmethod
    def step(state):
        ''' Predict the action given the current state.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
        '''
        #return np.random.choice(list(state['legal_actions'].keys()))
        return np.random.choice(state['raw_legal_actions'])     # i.e. 'raise' / 'check' etc


    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation. 
            Used if the state is known, otherwise step() is used
            

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent
            probs (list): The list of action probabilities
        '''

        obs, legal_actions = str(state['raw_obs']), list(state['legal_actions'].keys())
        if obs not in self.V:
            # self.random_choices += 1
            return self.step(state), {}
        best_action_num = self.V[obs][1]
        best_action = self.get_action(best_action_num)
        if best_action in state['raw_legal_actions']:   # if our best action for this state is available take it
            # self.value_choices += 1
            return best_action, {}
        else:                                           # play randomly
            return self.step(state),{}



    def get_action(self, num):
        '''Given the action num return the action that is decoded e.g. "fold", "check"
        '''
        if num == 0:
            return 'call'
        elif num == 1:
            return 'raise'
        elif num == 2:
            return 'fold'
        elif num == 3:
            return 'check'
        else:
            raise Exception("Unrecognised action")



    
    def update_P(self, obs, legal_actions):
        '''
        For State Space P:
            1) add new state and actions for it, or
            2) update list of legal actions for existing state (add actions that are not already in the list)
                    
        '''
        # if existing state
        if obs in self.P.keys():
            for action in legal_actions:            # check all listed actions that can be done in this state
                if action not in self.P[obs].keys():    # if any not listed in P so far add it
                    # initialize now will change later
                    #self.P[obs][action] = [[0,0,0,0]]   # {next_st: [prob_next_st, rew_next_st, num_visited_next_st]}
                    self.P[obs][action] =[{},0] # so far zero times 
        else:
            for action in legal_actions:
                self.P[obs][action] =[{},0] # so far zero times 

    
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
        # return state['obs'].tostring(), list(state['legal_actions'].keys())
        return str(state['raw_obs']), list(state['legal_actions'].keys())


