import numpy as np
import random
import torch

from collections import namedtuple
from model import Model

'''
    Actions:
    |A| -> 4
    
    0 - walk forward
    1 - walk backward
    2 - turn left
    3 - turn right

    States:
    |S| -> 37
    
    Agent's velocity, along with ray-based perception of objects around agent's forward direction. 
    
    Rewards:
    +1 is provided for yellow banana and -1 is provided for blue banana.
'''

Experience = namedtuple('Experience', 'state, action, reward, next_state, done')

class Agent:
    # Initialize the constants here, create the function approximators and memory replay structure
    def __init__(self, eps, min_eps, eps_decay, lr, gamma, batch_size, state_size, action_size):
        self.eps = eps
        self.min_eps = min_eps
        self.eps_decay - eps_decay

        self.lr = lr
        self.gamma = gamma

        self.state_size = state_size
        self.action_size = action_size

        self.behavioral_model = Model(state_size, action_size)
        self.target_model = Model(state_size, action_size)

        self.batch_size = batch_size
        self.memory = Memory(n=15000, batch_size=batch_size)

    # Update the memory replay and exercise a eps-greedy action, update eps
    def step(self, state, action, reward, next_state, done):
        self.eps *= self.eps_decay
        self.eps = max(self.eps, self.min_eps)

        experience = Experience(state, action, reward, next_state, done)
        self.memory.add(experience)

    # Sample from memory replay, learn and update target every C steps
    def act(self):
        if self.C % self.nsteps == 0:
            if self.batch_size > len(self.memory):
                experience = self.memory.sample()
                self.learn(experience)

    # Update target model from the behavioral model
    def _learn(self, experience):
        pass

    def _greedy_sample(self, state):
        prob = np.random.uniform(0.0, 1.0)
        if prob < self.eps:
            return np.random.choice(np.arange(self.action_size))
        else:
            with torch.no_grad():
                state = torch.from_numpy(state).astype(torch.float32)
                outputs = self.behavioral_model(state)
            action = torch.argmax(outputs, dim=1).item()
            return action

    def _soft_update_policy(self, tau):
        for b_param, t_param in zip(self.behavioral_model.parameters(), self.target_model.parameters()):
            t_param.data.copy_(tau * b_param.data + (1.0 + tau) * t_param.data)

# TODO: prioritized memory replay (according to paper)
class Memory:
    def __init__(self, n):
        pass