import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
from model import Actor, Critic
from noise import OUNoise, OUNoiseSampler

Experience = namedtuple('Experience', 'state, action, reward, next_state, done')

class ActorPair:
    def __init__(self, learnt_actor, target_actor):
        self.learnt_actor = learnt_actor
        self.target_actor = target_actor

# Off-policy, actor-critic
class Agent:
    def __init__(self, gamma, actor_lr, critic_lr, batch_size, state_size, action_size, memory_size, num_agents):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device}')

        self.steps = 0
        self.nsteps = 20

        self.gamma = gamma
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents

        # Learns argmax_a[Q(s, a); theta_mu] = mu(s, a; theta_mu)
        #self.learnt_actor = Actor(state_size, action_size).to(self.device) # learnt
        #self.target_actor = Actor(state_size, action_size).to(self.device) # soft-update tracking

        self.actors = []
        for _ in range(num_agents):
            pair = ActorPair(Actor(state_size, action_size).to(self.device), Actor(state_size, action_size).to(self.device))
            self.actors.append(pair)

        # Learns to evaluate Q(s, mu(s, a); theta_q)
        self.learnt_critic = Critic(state_size, action_size).to(self.device) # learnt
        self.target_critic = Critic(state_size, action_size).to(self.device) # soft-update tracking

        # Optimizers
        self.actor_optim = optim.Adam(self.learnt_actor.parameters(), lr=actor_lr)
        self.critic_optim = optim.Adam(self.learnt_critic.parameters(), lr=critic_lr, weight_decay=0.0001)

        # Note: Could be replaced by parallel env batching
        self.batch_size = batch_size
        self.memory = Memory(memory_size, batch_size, 505)
        self.memory.to(self.device)

        # Soft-update
        self.tau = 1e-3

        # Noise
        #self.noise = OUNoiseSampler(action_size, 505)
        self.noise = OUNoise(action_size, 505)

    def reset(self):
        self.noise.reset()

    def step(self, states):
        self.learnt_actor.eval()
        with torch.no_grad():
            states = torch.tensor(states, dtype=torch.float32, device=self.device)
            actions = self.learnt_actor(states).cpu().data.numpy()
            actions += self.noise.sample()
            actions = np.clip(actions, -1, 1)
        self.learnt_actor.train()
        return actions

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
        if self.steps % self.nsteps == 0:
            if len(self.memory) > self.batch_size:
                for _ in range(10):
                    self.__learn()
            self.steps = 0

    def __learn(self):

        # Train critic using r + gamma * V(s)
        states, actions, rewards, next_states, dones = self.memory.sample()


        # Critic ------------------------------------------------------------------------------------------------------------------------------
        # We want the most probable actions for St (continuous space)
        best_next_actions = self.target_actor(next_states)
        q_targets = rewards + self.gamma * self.target_critic(next_states, best_next_actions) * (1 - dones)

        q_predictions = self.learnt_critic(states, actions)

        self.critic_optim.zero_grad()
        critic_loss = F.mse_loss(q_predictions, q_targets)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.learnt_critic.parameters(), 1)
        self.critic_optim.step()
        # Critic ------------------------------------------------------------------------------------------------------------------------------



        # Actor ------------------------------------------------------------------------------------------------------------------------------
        # Generate advantage and train actor (TD)
        best_current_actions = self.target_actor(states)
        advantage = -self.learnt_critic(states, best_current_actions).mean()
        
        self.actor_optim.zero_grad()
        advantage.backward()
        self.actor_optim.step()
        # Actor ------------------------------------------------------------------------------------------------------------------------------


        self.soft_update(self.learnt_actor, self.target_actor, self.tau)
        self.soft_update(self.learnt_critic, self.target_critic, self.tau)

    def soft_update(self, learnt, target, tau):
        for learnt_param, target_param in zip(learnt.parameters(), target.parameters()):
            target_param.data.copy_(tau * learnt_param.data + (1.0 - tau) * target_param.data)

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
        actions = torch.FloatTensor([ entry.action for entry in samples ])
        rewards = torch.FloatTensor([ entry.reward for entry in samples ]).unsqueeze(1)
        next_states = torch.FloatTensor([ entry.next_state for entry in samples ])
        dones = torch.Tensor([ entry.done for entry in samples ]).unsqueeze(1)

        return [ states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device) ]

    def __len__(self):
        return len(self.ring_buffer)
