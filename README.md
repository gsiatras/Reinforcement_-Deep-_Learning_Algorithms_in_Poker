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

### Double DQN Agent: double_dqn_agent(DoubleDQNAgent):
Network architecture:
![Blank diagram (1)](https://github.com/gsiatras/Reinforcement_Deep_Learning_Algorithms_in_Poker/assets/94067900/563249a3-21c4-47cf-be95-d7ce58af7f8c)


### Dueling Double DQN Agent: dueling_double_dqn_agent(DDDQNAgentV2)
Network architecture:
![Blank diagram (2)](https://github.com/gsiatras/Reinforcement_Deep_Learning_Algorithms_in_Poker/assets/94067900/7cf3ff8a-7c51-438a-9f52-4ba7dab4bed7)

### State Represatation used (inspired by Alpha Holdem):
![Blank diagram](https://github.com/gsiatras/Reinforcement_Deep_Learning_Algorithms_in_Poker/assets/94067900/b310908a-9e67-4716-9622-b21a7e70634f)

### Testing results vs Bluf Thresholf model ( model desinged to train agents ):

![fig](https://github.com/gsiatras/Reinforcement_Deep_Learning_Algorithms_in_Poker/assets/94067900/ef9ff892-b227-486e-bce3-4806b803fc12)

Currently working on optimizing our models and later on adding convolutional networks and prioritized experience replay. 




