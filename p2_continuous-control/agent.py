import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from model import Actor, Critic
from noise import OUNoise

Experience = namedtuple('Experience', 'state, action, reward, next_state, done')

LEARN_EVERY = 20
ITERS = 10

class DDPGAgent:
    '''Class representing the DDPG algorithm'''
    def __init__(self, state_size, action_size, config):
        '''Class constructor and parameters initialization'''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')

        self.timestep = 0

        seed = config['seed']

        self.gamma = config['gamma']
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = config['number_agents']

        # Learns argmax_a[Q(s, a); theta_mu] = mu(s, a; theta_mu)
        self.learnt_actor = Actor(seed, state_size, action_size).to(self.device) # learnt
        self.target_actor = Actor(seed, state_size, action_size).to(self.device) # soft-update tracking
        self.actor_optim = optim.Adam(self.learnt_actor.parameters(), lr=config['actor_lr'])

        # Learns to evaluate Q(s, mu(s, a); theta_q)
        self.learnt_critic = Critic(seed, state_size, action_size, 1).to(self.device) # learnt
        self.target_critic = Critic(seed, state_size, action_size, 1).to(self.device) # soft-update tracking
        self.critic_optim = optim.Adam(self.learnt_critic.parameters(), lr=config['critic_lr'])

        print(f'Summary:\nActor network:\n{self.learnt_actor}\nCritic network:\n{self.learnt_critic}')

        # Note: Could be replaced by parallel env batching
        self.batch_size = config['batch_size']
        self.memory = Memory(config['memory_size'], self.batch_size, seed)
        self.memory.to_device(self.device)

        # Soft-update
        self.tau = config['tau']

        # Noise
        self.noise = OUNoise(action_size, seed)
        self.noise_decay = config['noise_decay']

    def reset(self):
        '''Reset the noise state'''
        self.noise.reset()

    def act(self, states):
        '''Sample an action from the policy'''
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        self.learnt_actor.eval()
        with torch.no_grad():
            actions = self.learnt_actor(states).cpu().data.numpy()
        self.learnt_actor.train()

        actions += self.noise_decay * self.noise.sample()

        return np.clip(actions, -1, 1)

    def remember(self, states, actions, rewards, next_states, dones):
        '''Populates the replay memory with new batch of data'''
        n = len(states)

        assert(n == len(actions))
        assert(n == len(rewards))
        assert(n == len(next_states))
        assert(n == len(dones))

        for (state, action, reward, next_state, done) in zip(states, actions, rewards, next_states, dones):
            self.memory.add(Experience(state, action, reward, next_state, done))

    def step(self, timestep):
        '''Wraps and controls the training of the function approximators using soft-updating'''
        if len(self.memory) > self.batch_size and self.timestep % LEARN_EVERY == 0:
            for _ in range(ITERS):
                states, actions, rewards, next_states, dones = self.memory.sample()
                self.__learn(states, actions, rewards, next_states, dones)

    def __learn(self, states, actions, rewards, next_states, dones):
        '''Optimizes the function apprximators and soft-updates'''

        self.__optimize_critic(states, actions, rewards, next_states, dones)

        self.__optimize_actor(states)

        self.__soft_update(self.learnt_actor, self.target_actor, self.tau)
        self.__soft_update(self.learnt_critic, self.target_critic, self.tau)

        self.noise_decay *= self.noise_decay
        self.reset()

    def __optimize_critic(self, states, actions, rewards, next_states, dones):
        '''Optimizes the critic approximator'''
        best_next_actions = self.target_actor(next_states)
        q_targets = rewards + self.gamma * self.target_critic(next_states, best_next_actions) * (1 - dones)

        q_predictions = self.learnt_critic(states, actions)

        self.critic_optim.zero_grad()
        critic_loss = F.mse_loss(q_predictions, q_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnt_critic.parameters(), 1)
        self.critic_optim.step()

    def __optimize_actor(self, states):
        '''Optimizes the actor approximator'''
        best_current_actions = self.learnt_actor(states)
        advantage = -self.learnt_critic(states, best_current_actions).mean()
        
        self.actor_optim.zero_grad()
        advantage.backward()
        self.actor_optim.step()

    def __soft_update(self, learnt, target, tau):
        '''Soft-updates the target parameters'''
        for learnt_param, target_param in zip(learnt.parameters(), target.parameters()):
            target_param.data.copy_(tau * learnt_param.data + (1.0 - tau) * target_param.data)

class Memory:
    def __init__(self, n, batch_size, seed):
        '''Class contructor of replay memory'''
        random.seed(seed)

        self.n = n
        self.batch_size = batch_size
        self.ring_buffer = []
        self.current_i = 0

    def to_device(self, device):
        '''Sets the device as CPU or GPU'''
        self.device = device

    def add(self, experience):
        '''Adds new batch of data to ring buffer'''
        if len(self.ring_buffer) < self.n:
            self.ring_buffer.append(experience)
        else:
            self.current_i %= self.n
            self.ring_buffer[int(self.current_i)] = experience
        self.current_i += 1

    def sample(self):
        '''Samples random data'''
        samples = random.sample(self.ring_buffer, self.batch_size)
        states = torch.FloatTensor([ entry.state for entry in samples if entry is not None ])
        actions = torch.FloatTensor([ entry.action for entry in samples if entry is not None ]).squeeze(1)
        rewards = torch.FloatTensor([ entry.reward for entry in samples if entry is not None ]).unsqueeze(1)
        next_states = torch.FloatTensor([ entry.next_state for entry in samples if entry is not None ])
        dones = torch.Tensor([ entry.done for entry in samples if entry is not None ]).unsqueeze(1)

        return [ states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device) ]

    def __len__(self):
        '''Returns the length of the ring buffer'''
        return len(self.ring_buffer)
