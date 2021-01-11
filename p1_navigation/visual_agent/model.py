import numpy as np
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, seed, w, h, input_channels, action_size):
        super(Model, self).__init__()

        self.seed = torch.manual_seed(seed)

        # 1x52x52
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, (3, 3), (1, 1)),
            # 32x50x50
            nn.Conv2d(32, 64, (3, 3), (1, 1)),
            # 64x48x48
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),

            # 64x24x24
            nn.Conv2d(64, 32, (3, 3), (1, 1)),
            # 32x22x22
            nn.BatchNorm2d(),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2)
            # 32x11x11
        )

        self.fc = nn.Sequential(
            nn.Linear(),
            nn.BatchNorm1d(),
            nn.LeakyReLU(),
            nn.Linear()
        )

    def forward(self, state):
        return
