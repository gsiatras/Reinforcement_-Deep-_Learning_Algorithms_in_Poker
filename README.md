# Creating Reinforcement Learning agents on rlcard enviroment		

# New limit holdem game
A limit holdem mode with shorter deck 4x(A, 10, J, Q, K), 1 hand card, 2 public cards		
Purpose: Shorter state space, test simplier algorithms		
Threshold Agent and Threshold Agent2:		
Rule based models betting only on high cards and combinations
new_limit_holdem_human: play againt any suitable agent
		
# Algorithms Implemented
Q-learning variation for new limit holdem: ql_agent(QLAgent)
Policy Iteration for new limit holdem: pi_agent(PIAgent)		
SARSA variation algorithm for new limit holdem: sarsa_agent(SARSAAgent)		
Double DQN for limit holdem implemented: my_dqn_agent(MYDQNAgent)		
Double Dueling DQN for limit holdem implemented: my_dqn_v2(MYDQNAgentV2)


