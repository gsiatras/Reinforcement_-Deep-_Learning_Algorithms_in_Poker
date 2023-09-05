''' A toy example of playing against a random agent on Limit Hold'em
'''

import rlcard
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.agents import RandomAgent, MYDQNAgent
from rlcard.utils.utils import print_card
import os
# Make environment
env = rlcard.make('limit-holdem')
human_agent = HumanAgent(env.num_actions)
human_agent2 = HumanAgent(env.num_actions)
agent_0 = RandomAgent(num_actions=env.num_actions)

dqagent = MYDQNAgent(
        env=env,
        model_path=os.path.join(
            'experiments/new_limit_holdem_dqn_result/',
            'my_dqn_model',
        ),
        epsilon_decay=0.999,
        epsilon_start=1.0,
        epsilon_end=0.05,
        card_obs_shape=(6, 4, 13),
        action_obs_shape=(24, 3, 4),
        learning_rate=0.00005,
        num_actions=env.num_actions,
        batch_size=128,
        tgt_update_freq=1000,
        train_steps=1,
        buffer_size=1000,
        device=None
    )

agent1 = dqagent.load(
    model_path=os.path.join('experiments/new_limit_holdem_dqn_result/', 'my_dqn_model'),
)
if agent1 is not None:
    dqagent = agent1

env.set_agents([
    human_agent2,
    dqagent,
])

print(">> Limit Hold'em random agent")

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

    input("Press any key to continue...")
