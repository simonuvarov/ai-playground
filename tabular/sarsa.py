import logging

import gym
import numpy as np

from agent import Agent
from utils import random_argmax

logging.basicConfig(level=logging.INFO)


class SARSA_Agent(Agent):

    def update_Q_table(self, state, action, reward,  next_state):
        # choose the next action using soft policy
        next_action = self.epsilon_greedy_policy(next_state)

        self.Q[state, action] = self.Q[state, action] + self.config['alpha'] * \
            (reward + self.config['gamma'] * self.Q[next_state,
                                                    next_action] - self.Q[state, action])

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
                action = self.epsilon_greedy_policy(state)

                # get the next state and reward
                next_state, reward, done, _ = self.env.step(action)

                # add the transition to the trajectory
                trajectory.append((state, action, reward))

                # update the value in Q table
                self.update_Q_table(state, action, reward, next_state)

                # update the state
                state = next_state

                if done:
                    break

            # add the trajectory to the history
            self.save_trajectory(trajectory)

        logging.info('Training finished')

    def policy(self, obs):
        '''
        Generate a policy from the Q table.
        For SARSA, this is the same as epsilon-greedy policy.
        '''
        return self.epsilon_greedy_policy(obs)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    config = {'alpha': 0.001,           # learning rate
              'gamma': 0.9,             # discount factor
              'epsilon': 0.025,         # probability of picking a random action
              'epsilon_decay': 0.999,   # decay rate of epsilon
              'episode_count': 5000     # number of episodes to train,
              }

    agent = SARSA_Agent(env, config)
    agent.train()
    agent.run_policy_once()

    env.close()
