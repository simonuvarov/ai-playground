import numpy as np

from tabular.history import History
from tabular.utils import random_argmax


class Agent():
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, epsilon_decay=0.99):
        '''
        Initialize the agent
        env: the environment
        alpha: learning rate
        gamma: discount factor
        epsilon: probability of picking a random action
        epsilon_decay: decay rate of epsilon       
        '''
        self.env = env
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        # list of trajectories
        self.history = History()

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def epsilon_greedy_policy(self, obs):
        action = -1
        if np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = random_argmax(self.Q[obs])

        self.decay_epsilon()

        return action

    def update_Q_table(self, state, action, reward, next_state):
        '''
        Update the Q table
        '''
        pass

    def save_trajectory(self, trajectory):
        '''
        Save the trajectory metadata
        trajectory: list of tuples (state, action, reward)
        '''
        return self.history.append(trajectory)

    def train(self, episode_count=1000):
        pass

    def policy(self, obs):
        '''
        Return the action to take given the observation
        '''
        pass

    def run_policy(self, n=10):
        '''
        Run the policy n times
        '''
        for run in range(n):
            obs = self.env.reset()
            for step in range(100):
                self.env.render()
                action = self.policy(obs)
                obs, _, done, _ = self.env.step(action)
                if done:
                    self.env.render()
                    break
