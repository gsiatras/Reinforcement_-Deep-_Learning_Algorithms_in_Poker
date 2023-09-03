"""An example of solving Limit Hold'em with DQN
"""

import os
import argparse

import rlcard
from rlcard.agents import (
    MYDQNAgent,
    RandomAgent
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
        'limit-holdem',
        config={
            'seed': 0,
            'allow_step_back': True,
        }
    )
    eval_env = rlcard.make(
        'limit-holdem',
        config={
            'seed': 0,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize training Agent
    agent = MYDQNAgent(
        env=env,
        model_path=os.path.join(
            args.log_dir,
            'ql_model',
        ),
        epsilon_decay=0.995,
        epsilon_start=1.0,
        epsilon_end=0.01,
        card_obs_shape=(6, 4, 13),
        action_obs_shape=(24, 3, 4),
        learning_rate=0.00005,
        num_actions=env.num_actions,
        batch_size=64,
        tgt_update_freq=500,
        train_steps=1,
        device=None
    )
    #agent.load()  # If we have saved model, we first load the model

    # Evaluate Ql
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
        for episode in range(args.num_episodes):
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            agent.train()

            if episode % args.evaluate_every == 0:
                reward, winrate = tournament(
                    eval_env,
                    args.num_eval_games
                )
                logger.log_performance2(
                    episode,
                    reward[0],
                    winrate[0],
                    agent.get_avg_loss()
                )
                # print(agent.epsilon)
                #print(agent.v)


        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
        csv2_path, fig2_path = logger.csv2_path, logger.fig2_path
        csv3_path, fig3_path = logger.csv3_path, logger.fig3_path
    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'DQN rewards')
    plot_curve2(csv2_path, fig2_path, 'DQN winrate', 'winrate')
    plot_curve2(csv3_path, fig3_path, 'DQN avg_loss', 'avg_loss')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN Agent example in RLCard")
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/new_limit_holdem_dqn_result/',
    )

    args = parser.parse_args()

    train(args)