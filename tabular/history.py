import csv

import pandas as pd


class History ():
    def __init__(self):
        self.history = []

    def append(self, trajectory):
        self.history.append(trajectory)

    def __len__(self):
        return len(self.history)

    @property
    def episode_rewards(self):
        '''
        Return the cumulative rewards for each episode
        '''
        return [sum(t[2] for t in trajectory) for trajectory in self.history]

    @property
    def episode_lengths(self):
        '''
        Return the length of each episode (number of transitions)
        '''
        return [len(trajectory) for trajectory in self.history]

    def to_csv(self, path):
        '''
        Save the history to a csv file
        path: path to save the csv file
        '''
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'length'])
            for episode, reward, length in zip(
                    range(len(self)), self.episode_rewards, self.episode_lengths):
                writer.writerow([episode, reward, length])

    def to_df(self):
        '''
        Return a pandas dataframe
        '''
        return pd.DataFrame({
            'episode': range(len(self)),
            'reward': self.episode_rewards,
            'length': self.episode_lengths
        })
