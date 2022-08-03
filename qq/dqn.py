# Simple DQN network that does not use convolutional layers.
# This is work in progress. It works worse than REINFORCE and takes more time to train.

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make('CartPole-v1')

EPISODE_COUNT = 200
EPSILON = 0.5
MINIMAL_EPSILON = 0.01
GAMMA = 1
BATCH_SIZE = 64
BUFFER_SIZE = 50000

class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 96)
        self.dropout = nn.Dropout(p=0.5)
        self.affine2 = nn.Linear(96, 2)
    
    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = self.dropout(x)
        return self.affine2(x)


class ReplayBuffer():
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, state,action,reward,next_state,done):
        if self.buffer == self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action,reward,next_state, done))

    def sample(self, batch_size):
        rnd_idxs = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[idx] for idx in rnd_idxs]


buffer = ReplayBuffer(capacity=BUFFER_SIZE)
dqn = DQN()
optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decay_epsilon():
    global EPSILON
    EPSILON = max(EPSILON - 2*(EPSILON - MINIMAL_EPSILON)/(EPISODE_COUNT), MINIMAL_EPSILON)

def optimize():
    '''
    Optimize the model for one step.
    '''
    # Sample a batch of transitions
    batch = buffer.sample(BATCH_SIZE)
    
    states, actions, rewards, next_states, dones = list(zip(*batch))

    # Convert to tensors
    states = torch.from_numpy(np.array(states)).float()
    actions = torch.from_numpy(np.array(actions)).long()
    rewards = torch.from_numpy(np.array(rewards)).float()
    next_states = torch.from_numpy(np.array(next_states)).float()
    dones = torch.from_numpy(np.array(dones)).float()

   
    # Compute the Q value
    q = dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    # print(q.shape)

    with torch.no_grad():
        q_next = dqn(next_states).max(1).values

        # I'm not sure why it does not work
        # when I remove (1 - dones)
        q_target = rewards + GAMMA * q_next * (1 - dones.float())



    loss =  F.mse_loss(q_target, q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  
    

moving_average = 10

for episode in range(0,EPISODE_COUNT):
    state, episode_reward= env.reset(), 0

    t = 0
    while True:
        if episode > EPISODE_COUNT - 20:
            EPSILON = 0
            env.render()
        

        q_values = dqn(torch.tensor(state, requires_grad=False, dtype=torch.float32).to(device))
        action = torch.argmax(q_values.detach()).item()

        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        buffer.push(state, action, reward, next_state, done)

        state = next_state

        if len(buffer) > 100:
            optimize()
            optimize()
        t += 1
        if done:
            break
            
    moving_average = 0.95 * moving_average + 0.05 * episode_reward

    if episode % 10 == 0:
        print(f"Episode: {episode}, Last Reward: {episode_reward}, Moving Average: {moving_average}, Epsilon: {EPSILON}")
    
    decay_epsilon()
