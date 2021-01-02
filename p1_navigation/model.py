import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, state_size, action_size):
        super(Model, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, action_size, bias=True)
        )

    def forward(self, state):
        return self.layers(state)
