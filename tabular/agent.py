import numpy as np

from utils import random_argmax


class Agent():
    def __init__(self, env, config):
        self.env = env
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.config = config
        # list of trajectories
        self.history = []

    def decay_epsilon(self):
        self.config['epsilon'] *= self.config['epsilon_decay']

    def epsilon_greedy_policy(self, obs):
        action = -1
        if np.random.random() < self.config['epsilon']:
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
        trajectory: list of tuples
        '''
        return self.history.append(trajectory)

    def train(self):
        pass

    def policy(self, obs):
        '''
        Return the action to take given the observation
        '''
        pass

    def run_policy_once(self):
        obs = self.env.reset()

        for _ in range(100):
            self.env.render()
            action = self.policy(obs)
            obs, _, done, _ = self.env.step(action)
            if done:
                self.env.render()
                break
