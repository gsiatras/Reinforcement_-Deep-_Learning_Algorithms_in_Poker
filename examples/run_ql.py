''' An example of solve New Hold'em with Q-learning
'''
import os
import argparse

import rlcard
from rlcard.agents import (
    QLAgent, SARSAAgent,
    RandomAgent, ThresholdAgent, ThresholdAgent2
)
from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
    plot_curve2
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
    agent = QLAgent(
        env,
        os.path.join(
            args.log_dir,
            'ql_model',
        ),
        0.05
    )
    agent.load()  # If we have saved model, we first load the model

    # Evaluate Ql
    eval_env.set_agents([
        agent,
        ThresholdAgent2(num_actions=env.num_actions),
    ])

    env.set_agents([
        agent,
        ThresholdAgent2(num_actions=env.num_actions),
    ])

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            if episode % args.evaluate_every == 0:
                agent.save()  # Save model
                reward, winrate = tournament(
                    eval_env,
                    args.num_eval_games
                )
                logger.log_performance1(
                    episode,
                    reward[0],
                    winrate[0]
                )
                # print(agent.epsilon)
                #print(agent.v)
            agent.train()


        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
        csv2_path, fig2_path = logger.csv2_path, logger.fig2_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'Q-learning')
    plot_curve2(csv2_path, fig2_path, 'Q-learning')

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
        default=6000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/new_limit_holdem_ql_result/',
    )

    args = parser.parse_args()

    train(args)
