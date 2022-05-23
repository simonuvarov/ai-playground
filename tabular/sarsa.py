import logging

import gym

from agent import Agent

logging.basicConfig(level=logging.INFO)


class SARSA_Agent(Agent):

    def update_Q_table(self, state, action, reward,  next_state):
        next_action = self.epsilon_greedy_policy(next_state)

        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * self.Q[next_state,
                                          next_action] - self.Q[state, action])

    def train(self, episode_count=3000):
        logging.info('Training started')
        for episode in range(episode_count):

            state, trajectory, done = self.env.reset(), [], False

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
        return self.epsilon_greedy_policy(obs)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    agent = SARSA_Agent(env)
    agent.train()
    agent.run_policy()

    env.close()
