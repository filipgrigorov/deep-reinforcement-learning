import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from model import Model

Experience = namedtuple('Experience', 'state, action, reward, next_state, done')

# DDPG, Off-policy, actor-critic
class DDPGAgent:
    def __init__(self, gamma, lr, batch_size, update_steps, state_size, action_size, memory_size, num_agents):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Learns argmax_a[Q(s, a); theta_mu] = mu(s, a; theta_mu)
        self.learnt_actor = Model(state_size, action_size).to(self.device) # learnt
        self.target_actor = Model(state_size, action_size).to(self.device) # soft-update tracking

        # Learns to evaluate Q(s, mu(s, a); theta_q)
        self.learnt_critic = Model(state_size, action_size).to(self.device) # learnt
        self.target_critic = Model(state_size, action_size).to(self.device) # soft-update tracking

        # Optimizers
        self.actor_optim = optim.Adam(self.learnt_actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.learnt_critic.parameters(), lr=lr)

        # Note: Could be replaced by parallel env batching
        self.batch_size = batch_size
        self.memory = Memory(memory_size, batch_size, 505)
        self.memory.to(self.device)

        # Soft-update
        self.update_counter = 0
        self.update_steps = update_steps
        self.tau = 1e-3

    def step(self, states):
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = self.learnt_actor(states)
            actions += self.noise(actions).float()
            return actions.cpu().numpy()

    def noise(self, x):
        theta = 0.15; sigma = 0.2
        return -theta * x + sigma * torch.rand(*x.size()).to(self.device)

    def remember(self, states, actions, rewards, next_states, dones):
        n = len(states)

        assert(n == len(actions))
        assert(n == len(rewards))
        assert(n == len(next_states))
        assert(n == len(dones))

        for idx in range(n):
            state = states[idx]
            action = actions[idx]
            reward = rewards[idx]
            next_state = next_states[idx]
            done = dones[idx]

            self.memory.add(Experience(state, action, reward, next_state, done))

    def act(self):
        if len(self.memory) <= self.batch_size:
            return

        # Train critic using r + gamma * V(s)
        states, actions, rewards, next_states, dones = self.memory.sample()

        best_actions = self.target_actor(next_states).argmax(dim=1).unsqueeze(0)
        q_outputs = self.learnt_critic(states)
        q_targets = rewards + self.gamma * self.target_critic(next_states).detach().gather(1, best_actions) * (1.0 - dones)

        # debug
        print(q_outputs.size())
        print(q_targets.size())

        raise('debug')

        self.critic_optim.zero_grad()
        critic_loss = F.mse_loss(q_targets, q_outputs)
        critic_loss.backward()
        self.critic_optim.step()

        # Generate advantage and train actor


        raise('debug')
        self.update_counter += 1
        if self.update_counter % self.update_steps == 0:
            self.soft_update(self.learnt_critic, self.target_critic)

class Memory:
    def __init__(self, n, batch_size, seed):
        self.seed = random.seed(seed)
        self.n = n
        self.batch_size = batch_size
        self.ring_buffer = []
        self.current_i = 0

    def to(self, device):
        self.device = device

    def add(self, experience):
        if len(self.ring_buffer) < self.n:
            self.ring_buffer.append(experience)
        else:
            self.current_i %= self.n
            self.ring_buffer[int(self.current_i)] = experience
        self.current_i += 1

    def sample(self):
        samples = random.sample(self.ring_buffer, self.batch_size)
        states = torch.FloatTensor([ entry.state for entry in samples ])
        actions = torch.LongTensor([ entry.action for entry in samples ]).unsqueeze(1)
        rewards = torch.FloatTensor([ entry.reward for entry in samples ]).unsqueeze(1)
        next_states = torch.FloatTensor([ entry.next_state for entry in samples ])
        dones = torch.Tensor([ entry.done for entry in samples ]).unsqueeze(1)

        return [ states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device) ]

    def __len__(self):
        return len(self.ring_buffer)
