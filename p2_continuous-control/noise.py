import copy
import numpy as np
import random
import torch
import torch.nn as nn

class OUNoise():
    '''Ornstein-Uhlenbeck process'''
    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)

        self.reset()

    def reset(self):
        '''Reset the internal state (= noise) to mean (mu)'''
        self.state = copy.copy(self.mu)

    def sample(self):
        '''Update internal state and return it as a noise sample'''
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.standard_normal(self.size)
        self.state += dx
        return self.state
