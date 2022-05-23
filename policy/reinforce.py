import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v1')


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 96)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(96, 2)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.dropout(x)
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=-1)


def select_action(state):
    state = torch.Tensor(state).unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample().item()

    return (action, probs[0][action])


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
running_reward = 10  # ~average reward for a random policy

for episode in range(1, 1000):
    state, history, loss = env.reset(), [], 0
    optimizer.zero_grad()

    while True:
        if episode > 500:
            env.render()

        # select action
        state = torch.Tensor(state)
        probs = policy(state)
        m = Categorical(probs)
        action_t = m.sample()  # required for log_prob
        action = action_t.item()

        state, reward, done, _ = env.step(action)

        history.append((action, m.log_prob(action_t), reward))

        if done:
            break

    # calculate loss
    G = 0
    for (action, log_prob, reward) in history[::-1]:
        G = G + reward
        loss -= log_prob * G
    loss.backward()
    optimizer.step()

    running_reward = 0.95*running_reward + 0.05*G
    if episode % 10 == 0:
        print(
            f"Episode {episode}, Last reward: {G}, Running reward: {running_reward}")
