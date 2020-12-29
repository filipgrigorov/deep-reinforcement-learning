import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

STEPS = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MSELoss(targets, predictions):
    return torch.mean(torch.square(targets - predictions))

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.seed = random.seed(seed)
        self.state_size = state_size
        self.action_size = action_size

        # Model: 
        # (i) Note: use ParallelData
        self.behaviour = QNetwork(state_size, action_size, seed).to(device)
        self.target = QNetwork(state_size, action_size, seed).to(device)

        self.optimizer = optim.Adam(self.behaviour.parameters(), lr=5e-4)

        # Hyperparameters of the agent
        self.batch_size = 64
        self.gamma = 0.99
        self.steps = 0

        # Replay memory (imitates the hypoccampus)``
        n = 10000
        self.memory = Memory(n, state_size)

    
    def step(self, state, action, reward, next_state, done):
        # Save the state in the replay memory
        experience = namedtuple('Experience', 'state action reward next_state')
        experience.state = state
        experience.action = action
        experience.reward = reward
        experience.next_state = next_state
        experience.dones = done

        self.memory.add(experience)

        # Sample from replay buffer (batch)
        if len(self.memory) > self.batch_size:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

            self.learn([states, actions, next_states, dones], self.gamma)


    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        prob = np.random.uniform(0.0, 1.0)
        if prob < eps:
            action = np.random.choice(np.arange(self.action_size))
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action = torch.argmax(self.behaviour(state)).item()
        return action


    def learn(self, experience, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """    

        states, actions, rewards, next_states, dones = experience

        # Forward pass according to DQN algorithm
        targets = rewards + gamma * self.target(next_states).max(1)[0].unsqueeze(1) * (1.0 - dones)
        
        # States from experiences batch
        outputs = self.behaviour(states).gather(1, actions)
        
        loss = F.mse_loss(outputs, targets)
        
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        # Call self.soft_update every C steps
        self.steps += 1
        if self.steps % STEPS == 0:
            self.soft_update(self.behaviour, self.target, tau=0.8)


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """

        # Note: Copy in place the data for each parameter
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        '''nparams = len(list(target_model.children()))
        for idx in range(0, nparams):
            target_model.layers[idx].weight.data = tau * local_model.layers[idx].weight.data + \
                (1.0 - tau) * target_model.layers[idx].weight.data
            target_model.layers[idx].bias.data = tau * local_model.layers[idx].bias.data + \
                (1.0 - tau) * target_model.layers[idx].bias.data'''

class Memory:
    def __init__(self, size, state_size):
        self.n = size
        self.state_size  = state_size
        self.ring_buffer = []
        self.idx = 0

    def add(self, experience):
        if len(self.ring_buffer) < self.n:
            self.ring_buffer.append(experience)
        else:
            self.idx %= self.n
            self.ring_buffer[self.idx] = experience
        self.idx += 1

    def sample(self, batch_size):
        batch_sample = random.sample(self.ring_buffer, batch_size)
        # Turn to torch tensors
        states = torch.from_numpy(batch_sample[:, 1]).to(device)
        actions = torch.from_numpy(batch_sample[:, 2]).to(device)
        rewards = torch.from_numpy(batch_sample[:, 3]).to(device)
        next_states = torch.from_numpy(batch_sample[:, 4]).to(device)
        dones = torch.from_numpy(batch_sample[:, 5]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.ring_buffer)

    def __str__(self):
        i = self.idx - 1
        str_repr = 'Last experience -> { '
        str_repr += str(self.ring_buffer[i].state) + ' '
        str_repr += str(self.ring_buffer[i].action) + ' '
        str_repr += str(self.ring_buffer[i].reward) + ' '
        str_repr += str(self.ring_buffer[i].next_state) + ''
        str_repr += str(self.ring_buffer[i].dones)
        str_repr += ' }'
        return str_repr

def test_ring_buffer():
    # tuple of (s, a, r, s', done)

    n = 5
    state_size = 8

    buf = Memory(5, 8)

    for i in range(0, 6):
        s = namedtuple('State', 'state, action, reward, next_state')
        s.state = np.random.choice(state_size, state_size)
        s.action = np.random.randint(0, 3)
        s.reward = round(np.random.uniform(0.0, 1.0), 2)
        s.next_state = np.random.choice(state_size, state_size)
        buf.add(s)

        print(len(buf))
        print(buf)

def test_soft_update():
    state_size = 8
    action_size = 4
    agent = Agent(state_size, action_size, seed=0)
    agent.soft_update(agent.target, agent.behaviour, tau=0.03)

if __name__ == '__main__':
    test_ring_buffer()
