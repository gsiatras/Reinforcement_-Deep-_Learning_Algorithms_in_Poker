''' An example of solve New Hold'em with Q-learning
'''
import os
import argparse

import rlcard
from rlcard.agents import (
    QLAgent, SARSAAgent,
    RandomAgent, ThresholdAgent, ThresholdAgent2, ThresholdAgent3
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
        'new-limit-holdem2',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'new-limit-holdem2',
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize training Agent
    agent = QLAgent(
        env,
        os.path.join(
            args.log_dir,
            'ql_model',
        ),
    )
    agent.load()  # If we have saved model, we first load the model

    # Evaluate Ql
    eval_env.set_agents([
        agent,
        ThresholdAgent3(num_actions=env.num_actions),
    ])

    env.set_agents([
        agent,
        ThresholdAgent3(num_actions=env.num_actions),
    ])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                agent.save()  # Save model
                logger.log_performance(
                    episode,
                    tournament(
                        eval_env,
                        args.num_eval_games
                    )[0]
                )
                # print(agent.epsilon)
                #print(agent.v)
            agent.train()


        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'Q-learning')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Q-Learning Agent example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=20000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=250,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/new_limit_holdem_ql_result/',
    )

    args = parser.parse_args()

    train(args)
