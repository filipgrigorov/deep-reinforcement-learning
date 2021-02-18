import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# input: state (33 real numbers)
# output action : [-1, 1]

# "Fan-in" is a term that defines the maximum number of inputs that a system can accept
# "Fan-out" is a term that defines the maximum number of inputs that the output of a system can feed to other systems

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    def __init__(self, seed, input_size, output_size):
        super(Actor, self).__init__()

        torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.Linear(400, 300),
            nn.Linear(300, output_size)
        )

        self.bn = nn.BatchNorm1d(400)

        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(len(self.fc) - 1):
            fc = self.fc[idx]
            fc.weight.data.uniform_(*hidden_init(fc))

        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        outputs = F.leaky_relu(self.bn(self.fc[0](states)))
        outputs = F.leaky_relu(self.fc[1](outputs))
        outputs = torch.tanh(self.fc[-1](outputs))
        return outputs

class Critic(nn.Module):
    def __init__(self, seed, input_size, action_size, output_size=1):
        super(Critic, self).__init__()

        torch.manual_seed(seed)

        self.fc = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.Linear(400 + action_size, 300),
            nn.Linear(300, output_size)
        )

        self.bn = nn.BatchNorm1d(400)

    def reset_parameters(self):
        for idx in range(len(self.fc) - 1):
            fc = self.fc[idx]
            fc.weight.data.uniform_(*hidden_init(fc))

        self.fc[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        outputs = F.leaky_relu(self.bn(self.fc[0](states)))
        outputs = torch.cat((outputs, actions), dim=1)
        outputs = F.leaky_relu(self.fc[1](outputs))
        return self.fc[2](outputs)
