# Creating Reinforcement Learning agents on rlcard enviroment		

# New limit holdem game
A limit holdem mode with shorter deck 4x(A, 10, J, Q, K), 1 hand card, 2 public cards		

Purpose: Shorter state space, test simplier algorithms		

Threshold Agent and Threshold Agent2:		

Rule based models betting only on high cards and combinations

new_limit_holdem_human: play againt any suitable agent
		
# Algorithms Implemented

## Phase 1 (new limit holdem):
### Q-learning variation algorithm: ql_agent(QLAgent)

### policy iteration algorithm: pi_agent(PIAGENT) 

### SARSA algorithm: sarsa_agent(SARSAAgent)

## Phase 2 (Full limit holdem game using Neural Networks):

### Double DQN Agent: double_dqn_agent(DoubleDQNAgent)

### Dueling Double DQN Agent: dueling_double_dqn_agent(DDDQNAgentV2)



