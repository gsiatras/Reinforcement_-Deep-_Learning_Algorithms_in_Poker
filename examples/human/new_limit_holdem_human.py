''' A toy example of playing against a random agent on new Limit Hold'em
'''
import os

# import args as args

import rlcard
from rlcard.agents import LimitholdemHumanAgent as HumanAgent, SARSAAgent
from rlcard.agents import RandomAgent, ThresholdAgent, ThresholdAgent2, QLAgent, PIAgent
from rlcard.utils.utils import print_card

# Make environment
env = rlcard.make('new-limit-holdem')
human_agent = HumanAgent(env.num_actions)
# Init agent:
agent_0 = RandomAgent(num_actions=env.num_actions)
agent_0 = ThresholdAgent(num_actions=env.num_actions)
agent_0 = ThresholdAgent2(num_actions=env.num_actions)

sarsa_agent = SARSAAgent(
        env,
        os.path.join(
            'experiments/new_limit_holdem_sarsa_result/sarsa_model',
        ),
    )
sarsa_agent.load()

ql_agent = QLAgent(
        env,
        os.path.join(
            'experiments/new_limit_holdem_ql_result/ql_model',
        ),
    )
ql_agent.load()


pi_agent = PIAgent(
        env,
        os.path.join(
            'experiments/new_limit_holdem_pi_result/pi_model',
        ),
)
pi_agent.load()

# hard code the agen you want to play against (pi_agent, ql_agent, sarsa_agent)
env.set_agents([
    human_agent,
    pi_agent,
])

print(">>New Limit Hold'em random agent")

while (True):
    print(">> Start a new game")

    trajectories, payoffs = env.run(is_training=False)
    # If the human does not take the final action, we need to
    # print other players action

    if len(trajectories[0]) != 0:
        final_state = trajectories[0][-1]
        action_record = final_state['action_record']
        state = final_state['raw_obs']
        _action_list = []
        for i in range(1, len(action_record)+1):
            """
            if action_record[-i][0] == state['current_player']:
                break
            """
            _action_list.insert(0, action_record[-i])
        for pair in _action_list:
            print('>> Player', pair[0], 'chooses', pair[1])

    # Let's take a look at what the agent card is
    print('=============     Random Agent    ============')
    print_card(env.get_perfect_information()['hand_cards'][1])

    print('===============     Result     ===============')
    if payoffs[0] > 0:
        print('You win {} chips!'.format(payoffs[0]))
    elif payoffs[0] == 0:
        print('It is a tie.')
    else:
        print('You lose {} chips!'.format(-payoffs[0]))
    print('')

    # print(trajectories[0])
    # print("=========================================================================================================")
    # print(trajectories[1])
    input("Press any key to continue...")
