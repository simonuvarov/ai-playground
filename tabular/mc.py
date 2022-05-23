import logging

import gym

from agent import Agent

logging.basicConfig(level=logging.INFO)


class MonteCarlo_Agent(Agent):
    def update_Q_table(self, trajectory):
        G = 0
        for (state, action, reward, _) in trajectory[::-1]:
            G = self.gamma * G + reward
            self.Q[state, action] += self.alpha * \
                (G - self.Q[state, action])

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

                state = next_state

                if done:
                    break

            self.update_Q_table(trajectory)

            self.save_trajectory(trajectory)

        logging.info('Training finished')

    def policy(self, obs):
        return self.epsilon_greedy_policy(obs)


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False, map_name='8x8')

    agent = MonteCarlo_Agent(env)
    agent.train()
    agent.run_policy()

    env.close()
