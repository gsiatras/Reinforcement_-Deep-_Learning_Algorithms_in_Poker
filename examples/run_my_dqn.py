"""An example of solving Limit Hold'em with DQN
"""

import os
import argparse

import rlcard
from rlcard.agents import (
    MYDQNAgent,
    RandomAgent,
    BluffAgent, MYDQNAgentV2, MYDQNAgentV3
)

from rlcard.utils import (
    set_seed,
    tournament,
    Logger,
    plot_curve,
    plot_curve2,
    plot_curve_multi
)

def train(args):
    # Make environments
    env1 = rlcard.make(
        'limit-holdem',
        config={
            'seed': 42,
            'allow_step_back': True,
        }
    )
    eval_env1 = rlcard.make(
        'limit-holdem',
        config={
            'seed': 42,
        }
    )

    env2 = rlcard.make(
        'limit-holdem',
        config={
            'seed': 42,
            'allow_step_back': True,
        }
    )
    eval_env2 = rlcard.make(
        'limit-holdem',
        config={
            'seed': 42,
        }
    )

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Initilize training Agent
    agent2 = MYDQNAgentV2(
        env=env2,
        model_path=os.path.join(
            args.log_dir,
            'my_dqn_model_complex',
        ),
        epsilon_decay=0.9999,
        epsilon_start=1.0,
        epsilon_end=0.05,
        card_obs_shape=(6, 4, 13),
        action_obs_shape=(24, 3, 4),
        learning_rate=0.00025,
        num_actions=env2.num_actions,
        batch_size=128,
        tgt_update_freq=10000,
        train_steps=1,
        buffer_size=100000,
        device=None
    )

    agent1 = MYDQNAgent(
        env=env1,
        model_path=os.path.join(
            args.log_dir,
            'my_dqn_model_complex',
        ),
        epsilon_decay=0.9999,
        epsilon_start=1.0,
        epsilon_end=0.05,
        card_obs_shape=(6, 4, 13),
        action_obs_shape=(24, 3, 4),
        learning_rate=0.00025,
        num_actions=env1.num_actions,
        batch_size=128,
        tgt_update_freq=10000,
        train_steps=1,
        buffer_size=100000,
        device=None
    )


    joker1 = BluffAgent(4, env1)
    joker_eval1 = BluffAgent(4, eval_env1)

    joker2 = BluffAgent(4, env2)
    joker_eval2 = BluffAgent(4, eval_env2)
    # Evaluate
    eval_env1.set_agents([
        agent1,
        joker_eval1
    ])

    env1.set_agents([
        agent1,
        joker1
    ])

    eval_env2.set_agents([
        agent1,
        joker_eval1
    ])

    env2.set_agents([
        agent1,
        joker1
    ])


    # Start training
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            print('\rIteration {}'.format(episode), end='')
            # Evaluate the performance. Play with Random agents.
            agent1.train()
            agent2.train()

            if episode % args.evaluate_every == 0:
                reward1, winrate1 = tournament(
                    eval_env1,
                    args.num_eval_games
                )
                reward2, winrate2 = tournament(
                    eval_env2,
                    args.num_eval_games
                )

                loss1, epsilon1 = agent1.get_avg_loss()
                loss2, epsilon2 = agent1.get_avg_loss()
                logger.log_performance_multi(
                    episode,
                    reward1[0],
                    winrate1[0],
                    loss1,
                    epsilon1,
                    reward2[0],
                    winrate2[0],
                    loss2,
                    epsilon2
                )
                # agent2.save()
                # print(agent.start_pos)
                # print(agent.epsilon)
                #print(agent.v)


        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path
        csv2_path, fig2_path = logger.csv2_path, logger.fig2_path
        csv3_path, fig3_path = logger.csv3_path, logger.fig3_path
        csv4_path, fig4_path = logger.csv4_path, logger.fig4_path
        csv11_path, fig11_path = logger.csv11_path, logger.fig11_path
        csv21_path, fig21_path = logger.csv21_path, logger.fig21_path
        csv31_path, fig31_path = logger.csv31_path, logger.fig31_path
        csv41_path, fig41_path = logger.csv41_path, logger.fig41_path
    # Plot the learning curve
    plot_curve_multi(csv_path, csv11_path, fig_path, 'Double DQN', 'Dueling Double DQN', 'reward')
    plot_curve_multi(csv2_path, csv21_path, fig2_path, 'Double DQN', 'Dueling Double DQN', 'winrate')
    plot_curve_multi(csv3_path, csv31_path, fig3_path, 'Double DQN', 'Dueling Double DQN', 'avg_loss')
    plot_curve_multi(csv4_path, csv41_path, fig4_path, 'Double DQN', 'Dueling Double DQN', 'epsilon')

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
        default=100001,
    )
    parser.add_argument(
        '--num_eval_games',
        type=int,
        default=3000,
    )
    parser.add_argument(
        '--evaluate_every',
        type=int,
        default=10000,
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='experiments/new_limit_holdem_dqn_result/',
    )

    args = parser.parse_args()

    train(args)