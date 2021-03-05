import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# input: states (24)
# output: actions [-1, 1]

# "Fan-in" is a term that defines the maximum number of inputs that a system can accept
# "Fan-out" is a term that defines the maximum number of inputs that the output of a system can feed to other systems

def generate_fc(input_size, hidden_in, hidden_out, output_size, action_size=None):
    return nn.Sequential(
        nn.Linear(input_size, hidden_in),
        nn.Linear(hidden_in if action_size is None else hidden_in + action_size, hidden_out),
        nn.Linear(hidden_out, output_size)
    ), nn.BatchNorm1d(hidden_in)

def hidden_init(layer):
    '''Computes the range of the randomization'''
    fan_in = layer.weight.data.size(0)
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    '''Actor network architecture'''
    def __init__(self, seed, input_size, output_size):
        '''Class constructor'''
        super(Actor, self).__init__()

        torch.manual_seed(seed)

        self.fc, self.bn = generate_fc(input_size, 400, 300, output_size)

        self.init_parameters()

    def init_parameters(self):
        '''Weights initialization'''
        for idx in range(len(self.fc) - 1):
            fc = self.fc[idx]
            fc.weight.data.uniform_(*hidden_init(fc))

        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        '''Forward pass override of the nn.Module'''
        outputs = F.leaky_relu(self.bn(self.fc[0](state)))
        outputs = F.leaky_relu(self.fc[1](outputs))
        outputs = self.fc[-1](outputs) #torch.tanh(self.fc[-1](outputs))
        return outputs

class Critic(nn.Module):
    '''Critic network architecture'''
    def __init__(self, seed, input_size, action_size, output_size=1):
        '''Class constructor'''
        super(Critic, self).__init__()

        torch.manual_seed(seed)

        self.fc, self.bn = generate_fc(input_size, 400, 300, output_size, action_size)

        self.init_parameters()

    def init_parameters(self):
        '''Weights initialization'''
        for idx in range(len(self.fc) - 1):
            fc = self.fc[idx]
            fc.weight.data.uniform_(*hidden_init(fc))

        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, actions):
        '''Forward pass override of the nn.Module'''
        outputs = F.leaky_relu(self.bn(self.fc[0](state)))
        outputs = torch.cat((outputs, actions), dim=1)
        outputs = F.leaky_relu(self.fc[1](outputs))
        return self.fc[2](outputs)
