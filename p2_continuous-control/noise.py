import copy
import numpy as np
import random
import torch
import torch.nn as nn

class OUNoise():
    '''Ornstein-Uhlenbeck process.'''

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        '''Initialize parameters and noise process.'''
        self.mu = mu * np.ones(size) # action size
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu).'''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''Update internal state and return it as a noise sample.'''
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        return self.state

class OUNoiseSampler:
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        torch.manual_seed(seed)

        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.drift = mu * torch.ones(size)

        self.reset()

    def reset(self):
        self.noise_state = copy.copy(self.drift)

    def sample(self):
        dx_dt = self.theta * (self.drift - self.noise_state) + self.sigma * torch.rand(self.size)
        self.noise_state += dx_dt
        return self.noise_state

if __name__ == '__main__':
    noise_sample_1 = OUNoise(4, 505)
    noise_sample_2 = OUNoiseSampler(4, 505)

    print(noise_sample_1.sample())
    print('------------------------------------------')
    print(noise_sample_2.sample())
