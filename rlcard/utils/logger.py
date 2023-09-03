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

    def log_performance2(self, episode, reward, winrate, avg_loss):
        ''' Log a point in the curve
        Args:
            episode (int): the episode of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'episode': episode, 'reward': reward})
        self.writer2.writerow({'episode': episode, 'winrate': winrate})
        self.writer3.writerow({'episode': episode, 'avg_loss': avg_loss})
        print('')
        self.log('----------------------------------------')
        self.log('  episode      |  ' + str(episode))
        self.log('  reward       |  ' + str(reward))
        self.log('  winrate      |  ' + str(winrate))
        self.log('  loss         |  ' + str(avg_loss))
        self.log('----------------------------------------')

    def __exit__(self, type, value, traceback):
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()
        if self.txt2_path is not None:
            self.txt2_file.close()
        if self.csv3_path is not None:
            self.csv2_file.close()
        if self.txt3_path is not None:
            self.txt3_file.close()
        if self.csv3_path is not None:
            self.csv3_file.close()
        print('\nLogs saved in', self.log_dir)
