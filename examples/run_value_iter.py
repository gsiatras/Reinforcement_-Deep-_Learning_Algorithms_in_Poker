'''
Example of running value iteration agent against known type of opponent.
This time threshold agent
'''

import os
import argparse

import rlcard
from rlcard.agents import (
    ThresholdAgent,
    ThresholdAgent2,
    RandomAgent,
)

from rlcard.agents.value_iteration_agent import ValueIterAgent
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
)


def train(args):
    # Make environments
    env = rlcard.make(
        'new-limit-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'new-limit-holdem',
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize training Agent
    agent = ValueIterAgent(
        env,
        os.path.join(
            args.log_dir,
            'vi_model',

        ),
    )
    #agent.load()  # If we have saved model, we first load the model

    # Evaluate Value Iteration
    eval_env.set_agents([
        agent,
        # ThresholdAgent(num_actions=env.num_actions),
        # RandomAgent(num_actions=env.num_actions)
        ThresholdAgent2(num_actions=env.num_actions)
    ])

    env.set_agents([
        agent,
        # ThresholdAgent(num_actions=env.num_actions),
        # RandomAgent(num_actions=env.num_actions)
        ThresholdAgent2(num_actions=env.num_actions)
    ])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            print('\rLearning the enviroment {}'.format(episode), end='')
            agent.learn_env()
            
            # now begin training
        agent.value_iteration_algo()
        for episode in range(args.num_eval_games):
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )
        # print('Random choices {}'.format(agent.random_choices))
        # print('Value choices {}'.format(agent.value_choices))
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'Value Iteration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Value Iteration Agent example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/new_limit_holdem_vi_result/',
    )
 
    args = parser.parse_args()

    train(args)