import logging

import gym
import numpy as np

from agent import Agent

logging.basicConfig(level=logging.INFO)


class MC_Agent(Agent):
    def calculate_return(self, trajectory):
        '''
        Calculate the return of the first state in a trajectory
        trajectory: list of tuples
        '''
        r = 0
        for index, transition in enumerate(trajectory):
            reward = transition[2]
            r += (self.gamma ** index) * reward
        return r

    def update_Q_table(self, trajectory):
        '''
        Update the Q table
        trajectory: list of tuples
        '''
        for index, (state, action, _) in enumerate(trajectory):
            G = self.calculate_return(trajectory[index:])
            self.Q[state, action] += self.alpha * \
                (G - self.Q[state, action])

    def train(self, episode_count=1000):
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
        return self.epsilon_greedy_policy(obs)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    agent = MC_Agent(env)
    agent.train(2000)
    agent.run_policy_once()

    env.close()
