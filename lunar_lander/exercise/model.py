import torch
import torch.nn as nn
import torch.nn.functional as F

'''
The state. Attributes:

s[0] is the horizontal coordinate
s[1] is the vertical coordinate
s[2] is the horizontal speed
s[3] is the vertical speed
s[4] is the angle
s[5] is the angular speed
s[6] 1 if first leg has contact, else 0
s[7] 1 if second leg has contact, else 0

Action is two floats [main engine, left-right engines].

Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off

'''

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()

        self.seed = torch.manual_seed(seed)
        
        # Note: Shallower networks diverge
        self.layers = nn.Sequential(
            nn.Linear(in_features=state_size, out_features=200, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=200, bias=True),
            nn.LeakyReLU(),
            nn.Linear(in_features=200, out_features=action_size, bias=True)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.layers(state)
