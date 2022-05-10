import logging

import gym
import numpy as np

from agent import Agent

logging.basicConfig(level=logging.INFO)


class Q_Agent(Agent):
    def update_Q_table(self, state, action, reward,  next_state):
        next_action = np.argmax(self.Q[next_state])

        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * self.Q[next_state,
                                          next_action] - self.Q[state, action])

    def train(self, episode_count=3000):
        logging.info('Training started')
        for episode in range(episode_count):
            state = self.env.reset()
            trajectory = []
            done = False

            if(episode % 100 == 0):
                logging.info(f'Running episode {episode}')

            while True:
                action = self.epsilon_greedy_policy(state)
                next_state, reward, done, _ = self.env.step(action)

                trajectory.append((state, action, reward, next_state))

                self.update_Q_table(state, action, reward, next_state)

                state = next_state

                if done:
                    break

            self.save_trajectory(trajectory)

        logging.info('Training finished')

    def policy(self, obs):
        return np.argmax(self.Q[obs])


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    agent = Q_Agent(env, epsilon_decay=0.9, epsilon=0.2)
    agent.train()
    agent.run_policy()

    env.close()
