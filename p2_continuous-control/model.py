import numpy as np
import torch
import torch.nn as nn

# input: state (33 real numbers)
# output action : [-1, 1]

# "Fan-in" is a term that defines the maximum number of inputs that a system can accept
# "Fan-out" is a term that defines the maximum number of inputs that the output of a system can feed to other systems

torch.manual_seed(505)

def init_weights_of(layer):
    fan_in = layer.weight.data.size(0)
    val = 1.0 / np.sqrt(fan_in)
    layer.weight.data.uniform_(-val, val)

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.BatchNorm1d(400),
            nn.SELU(),
            nn.Linear(400, 300),
            nn.BatchNorm1d(300),
            nn.SELU()
        )

        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            if isinstance(layer, nn.Linear):
                init_weights_of(layer)

        self.output_layer = nn.Linear(300, output_size)

        # Note: Initialize close to zero
        torch.nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)

    def forward(self, states):
        return torch.tanh(self.output_layer(self.layers(states)))

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 400),
            nn.BatchNorm1d(400),
            nn.SELU(),
            nn.Linear(400, 300),
            nn.BatchNorm1d(300),
            nn.SELU()
        )

        for idx in range(len(self.layers)):
            layer = self.layers[idx]
            if isinstance(layer, nn.Linear):
                init_weights_of(layer)

        self.output_layer = nn.Linear(300 + output_size, output_size)

        # Note: Initialize close to zero
        torch.nn.init.uniform_(self.output_layer.weight, -3e-3, 3e-3)

    def forward(self, states, actions):
        # Note: According to paper, actions are included at the last output layer
        outputs = self.layers(states)
        outputs = torch.cat((outputs, actions), dim=1)
        return self.output_layer(outputs)
