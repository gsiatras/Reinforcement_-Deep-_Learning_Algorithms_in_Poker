import os
import csv

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir

    def __enter__(self):
        self.txt_path = os.path.join(self.log_dir, 'log.txt')
        self.csv_path = os.path.join(self.log_dir, 'performance.csv')
        self.fig_path = os.path.join(self.log_dir, 'fig.png')

        self.txt2_path = os.path.join(self.log_dir, 'log2.txt')
        self.csv2_path = os.path.join(self.log_dir, 'performance2.csv')
        self.fig2_path = os.path.join(self.log_dir, 'fig2.png')

        self.txt3_path = os.path.join(self.log_dir, 'log3.txt')
        self.csv3_path = os.path.join(self.log_dir, 'performance3.csv')
        self.fig3_path = os.path.join(self.log_dir, 'fig3.png')

        self.txt4_path = os.path.join(self.log_dir, 'log4.txt')
        self.csv4_path = os.path.join(self.log_dir, 'performance4.csv')
        self.fig4_path = os.path.join(self.log_dir, 'fig4.png')

        self.txt11_path = os.path.join(self.log_dir, 'log11.txt')
        self.csv11_path = os.path.join(self.log_dir, 'performance11.csv')
        self.fig11_path = os.path.join(self.log_dir, 'fig11.png')

        self.txt21_path = os.path.join(self.log_dir, 'log21.txt')
        self.csv21_path = os.path.join(self.log_dir, 'performance21.csv')
        self.fig21_path = os.path.join(self.log_dir, 'fig21.png')

        self.txt31_path = os.path.join(self.log_dir, 'log31.txt')
        self.csv31_path = os.path.join(self.log_dir, 'performance31.csv')
        self.fig31_path = os.path.join(self.log_dir, 'fig31.png')

        self.txt41_path = os.path.join(self.log_dir, 'log41.txt')
        self.csv41_path = os.path.join(self.log_dir, 'performance41.csv')
        self.fig41_path = os.path.join(self.log_dir, 'fig41.png')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ['episode', 'reward']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.txt2_file = open(self.txt2_path, 'w')
        self.csv2_file = open(self.csv2_path, 'w')
        fieldnames = ['episode', 'winrate']
        self.writer2 = csv.DictWriter(self.csv2_file, fieldnames=fieldnames)
        self.writer2.writeheader()

        self.txt3_file = open(self.txt3_path, 'w')
        self.csv3_file = open(self.csv3_path, 'w')
        fieldnames = ['episode', 'avg_loss']
        self.writer3 = csv.DictWriter(self.csv3_file, fieldnames=fieldnames)
        self.writer3.writeheader()

        self.txt4_file = open(self.txt4_path, 'w')
        self.csv4_file = open(self.csv4_path, 'w')
        fieldnames = ['episode', 'epsilon']
        self.writer4 = csv.DictWriter(self.csv4_file, fieldnames=fieldnames)
        self.writer4.writeheader()

        self.txt11_file = open(self.txt11_path, 'w')
        self.csv11_file = open(self.csv11_path, 'w')
        fieldnames = ['episode', 'reward']
        self.writer11 = csv.DictWriter(self.csv11_file, fieldnames=fieldnames)
        self.writer11.writeheader()

        self.txt21_file = open(self.txt21_path, 'w')
        self.csv21_file = open(self.csv21_path, 'w')
        fieldnames = ['episode', 'winrate']
        self.writer21 = csv.DictWriter(self.csv21_file, fieldnames=fieldnames)
        self.writer21.writeheader()

        self.txt31_file = open(self.txt31_path, 'w')
        self.csv31_file = open(self.csv31_path, 'w')
        fieldnames = ['episode', 'avg_loss']
        self.writer31 = csv.DictWriter(self.csv31_file, fieldnames=fieldnames)
        self.writer31.writeheader()

        self.txt41_file = open(self.txt41_path, 'w')
        self.csv41_file = open(self.csv41_path, 'w')
        fieldnames = ['episode', 'epsilon']
        self.writer41 = csv.DictWriter(self.csv41_file, fieldnames=fieldnames)
        self.writer41.writeheader()
        return self

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, episode, reward):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')

    def log_performance1(self, episode, reward, winrate):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward})
        self.writer2.writerow({'episode': episode, 'winrate': winrate})
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('  winrate       |  ' + str(winrate))
        self.log('----------------------------------------')

    def log_performance2(self, episode, reward, winrate, avg_loss, epsilon):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward})
        self.writer2.writerow({'episode': episode, 'winrate': winrate})
        self.writer3.writerow({'episode': episode, 'avg_loss': avg_loss})
        self.writer4.writerow({'episode': episode, 'epsilon': epsilon})
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('  winrate      |  ' + str(winrate))
        self.log('  loss         |  ' + str(avg_loss))
        self.log('  epsilon      |  ' + str(epsilon))
        self.log('----------------------------------------')

    def log_performance_multi(self, episode, reward1, winrate1, avg_loss1, epsilon1, reward2, winrate2, avg_loss2,
                              epsilon2):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward1})
        self.writer2.writerow({'episode': episode, 'winrate': winrate1})
        self.writer3.writerow({'episode': episode, 'avg_loss': avg_loss1})
        self.writer4.writerow({'episode': episode, 'epsilon': epsilon1})

        self.writer11.writerow({'episode': episode, 'reward': reward2})
        self.writer21.writerow({'episode': episode, 'winrate': winrate2})
        self.writer31.writerow({'episode': episode, 'avg_loss': avg_loss2})
        self.writer41.writerow({'episode': episode, 'epsilon': epsilon2})

        # Log performance for algorithm 1
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward1))
        self.log('  winrate      |  ' + str(winrate1))
        self.log('  loss         |  ' + str(avg_loss1))
        self.log('  epsilon      |  ' + str(epsilon1))

        # Log performance for algorithm 2
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward2))
        self.log('  winrate      |  ' + str(winrate2))
        self.log('  loss         |  ' + str(avg_loss2))
        self.log('  epsilon      |  ' + str(epsilon2))
        self.log('----------------------------------------')

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
        if self.txt2_path is not None:
            self.txt2_file.close()
        if self.csv3_path is not None:
            self.csv3_file.close()
        if self.txt3_path is not None:
            self.txt3_file.close()
        if self.csv3_path is not None:
            self.csv3_file.close()
        if self.txt4_path is not None:
            self.txt4_file.close()
        if self.csv4_path is not None:
            self.csv4_file.close()

        if self.txt11_path is not None:
            self.txt11_file.close()
        if self.csv11_path is not None:
            self.csv11_file.close()
        if self.txt21_path is not None:
            self.txt21_file.close()
        if self.csv31_path is not None:
            self.csv31_file.close()
        if self.txt31_path is not None:
            self.txt31_file.close()
        if self.csv31_path is not None:
            self.csv31_file.close()
        if self.txt41_path is not None:
            self.txt41_file.close()
        if self.csv41_path is not None:
            self.csv41_file.close()
        print('\nLogs saved in', self.log_dir)
