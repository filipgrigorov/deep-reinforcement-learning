import torch
import torch.nn as nn

# input: state (33 real numbers)
# output action : [-1, 1]

# Note: For the numerical state space, we are to use fully-connected layers
def build_model(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 400),
        nn.BatchNorm1d(20),
        nn.SELU(),
        nn.Linear(400, 256),
        nn.BatchNorm1d(20),
        nn.SELU(),
        nn.Linear(256, output_size)
    )

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()

        torch.manual_seed(505)

        self.layers = build_model(input_size=input_size, output_size=output_size)

    def forward(self, x):
        x = self.layers(x)
        # Note: Predict the actions fropm the state, directly
        return torch.tanh(x)

class Critic(nn.Module):
    def __init__(self, input_size, output_size):
        super(Critic, self).__init__()

        torch.manual_seed(505)

        self.layers = build_model(input_size, output_size)

    def forward(self, x):
        x = self.layers(x)
        return x
