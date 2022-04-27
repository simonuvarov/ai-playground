import logging

import gym
import numpy as np

from utils import random_argmax

logging.basicConfig(level=logging.INFO)


class Agent():
    def __init__(self, env, config):
        self.env = env
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.config = config
        # list of trajectories
        self.history = []

    def __decay_epsilon(self):
        self.config['epsilon'] *= self.config['epsilon_decay']

    def __epsilon_greedy_policy(self, obs):
        action = -1
        if np.random.random() < self.config['epsilon']:
            action = self.env.action_space.sample()
        else:
            action = random_argmax(self.Q[obs])

        self.__decay_epsilon()

        return action

    def calculate_return(self, trajectory):
        '''
        Calculate the return of the first state in a trajectory
        trajectory: list of tuples
        '''
        r = 0
        for index, transition in enumerate(trajectory):
            reward = transition[2]
            r += (self.config['gamma'] ** index) * reward
        return r

    def update_Q_table(self, trajectory):
        '''
        Update the Q table
        trajectory: list of tuples
        '''
        for index, (state, action, _) in enumerate(trajectory):
            G = self.calculate_return(trajectory[index:])
            self.Q[state, action] += self.config['alpha'] * \
                (G - self.Q[state, action])

    def save_trajectory(self, trajectory):
        '''
        Save the trajectory metadata
        trajectory: list of tuples
        '''
        return self.history.append(trajectory)

    def train(self):
        logging.info('Training started')
        for episode in range(self.config['episode_count']):

            # reset the episode
            state = self.env.reset()
            trajectory = []
            done = False

            # print the episode number
            if(episode % 500 == 0):
                logging.info(f'Running episode {episode}')

            while True:
                # pick action acoring to the epsilon-greedy policy
                action = self.__epsilon_greedy_policy(state)

                # get the next state and reward
                next_state, reward, done, _ = self.env.step(action)

                # add the transition to the trajectory
                trajectory.append((state, action, reward))

                # update the state
                state = next_state

                if done:
                    break

            # update the Q table after finishing the episode
            self.update_Q_table(trajectory)

            # add the trajectory to the history
            self.save_trajectory(trajectory)
        logging.info('Training finished')

    def policy(self, obs):
        '''
        Generate a policy from the Q table
        For the MC algorithm, it is the same as epsilon-greedy policy.
        '''
        return self.__epsilon_greedy_policy(obs)

    def run_policy_once(self):
        obs = self.env.reset()

        for _ in range(100):
            self.env.render()
            action = self.policy(obs)
            obs, _, done, _ = self.env.step(action)
            if done:
                env.render()
                break


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    config = {'alpha': 0.001,           # learning rate
              'gamma': 0.9,             # discount factor
              'epsilon': 0.025,         # probability of picking a random action
              'epsilon_decay': 0.999,   # decay rate of epsilon
              'episode_count': 5000     # number of episodes to train,
              }

    agent = Agent(env, config)
    agent.train()
    agent.run_policy_once()

    env.close()
