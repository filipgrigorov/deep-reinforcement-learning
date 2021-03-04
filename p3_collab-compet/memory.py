import random
import torch

from collections import namedtuple

Experience = namedtuple('Experience', 'states, actions, rewards, next_states, dones')

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
        states = torch.FloatTensor([ entry.states for entry in samples if entry is not None ])
        actions = torch.FloatTensor([ entry.actions for entry in samples if entry is not None ]).squeeze(1)
        rewards = torch.FloatTensor([ entry.rewards for entry in samples if entry is not None ]).unsqueeze(1)
        next_states = torch.FloatTensor([ entry.next_states for entry in samples if entry is not None ])
        dones = torch.Tensor([ entry.dones for entry in samples if entry is not None ]).unsqueeze(1)

        return [ states.to(self.device), actions.to(self.device), rewards.to(self.device), next_states.to(self.device), dones.to(self.device) ]

    def __len__(self):
        '''Returns the length of the ring buffer'''
        return len(self.ring_buffer)
