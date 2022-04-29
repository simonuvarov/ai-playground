import logging

import gym
import numpy as np

from agent import Agent

logging.basicConfig(level=logging.INFO)


class Q_Agent(Agent):
    def update_Q_table(self, state, action, reward,  next_state):
        '''
        Update the Q table
        '''
        # choose the next action greedy
        next_action = np.argmax(self.Q[next_state])

        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * self.Q[next_state,
                                          next_action] - self.Q[state, action])

    def train(self, episode_count=3000):
        logging.info('Training started')
        for episode in range(episode_count):

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
                trajectory.append((state, action, reward, next_state))

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
        For Q-learning, it means taking action with the highest Q value.
        '''
        return np.argmax(self.Q[obs])


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    # we don't need to decay epsilon for Q-learning
    agent = Q_Agent(env, epsilon_decay=0.9, epsilon=0.2)
    agent.train()
    agent.run_policy()

    env.close()
