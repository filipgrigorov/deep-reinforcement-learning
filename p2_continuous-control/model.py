import torch
import torch.nn as nn

# input: state (33 real numbers)
# output action : [-1, 1]

torch.manual_seed(505)

# Note: For the numerical state space, we are to use fully-connected layers
def build_model(input_size, output_size):
    return nn.Sequential(
        nn.Linear(input_size, 400),
        nn.BatchNorm1d(400),
        nn.SELU(),
        nn.Linear(400, 256),
        nn.BatchNorm1d(256),
        nn.SELU(),
        nn.Linear(256, output_size)
    )

class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()

        self.layers = build_model(input_size=input_size, output_size=output_size)

    def forward(self, x):
        return self.layers(x)
