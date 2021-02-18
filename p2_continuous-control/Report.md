# Project 2: Continuous Control

![alt](algorithm_description.png)
### Architecture

Actor: 3 fully connected layers (`nn.Linear`) with a tanh activation on the output to bound the output between -1 and 1. There is one batch normalization layer after the first fully connected one (`nn.BatchNorm1d`).

Critic: 3 fully connected layers where the second fully connected layer accepts as input (output_fc1 + action). There is one batch normalization layer after the first fully connected one.

All other activation functions are `F.leaky_relu`, in order to avoid dead neurons when learning.
