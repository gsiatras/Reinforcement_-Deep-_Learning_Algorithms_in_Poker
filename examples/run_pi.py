''' An example of solving New Hold'em with policy iteration
'''
import os
import argparse

import rlcard
from rlcard.agents import (
    QLAgent, SARSAAgent,
    RandomAgent, ThresholdAgent, ThresholdAgent2, PIAgent
)
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
    agent = PIAgent(
        env,
        os.path.join(
            args.log_dir,
            'ql_model',
        ),
    )


    # Evaluate PI
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])

    env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])


    # Start training
    with Logger(args.log_dir) as logger:
        # Evaluate the performance.
        agent.train()
        agent.save()
        for episode in range(args.num_episodes):
            logger.log_performance(
                episode,
                tournament(
                    eval_env,
                    args.num_eval_games
                )[0]
            )
        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'Policy-iteration')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Policy Iteration Agent example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/new_limit_holdem_pi_result/',
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=20,
    )

    args = parser.parse_args()
    train(args)
