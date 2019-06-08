from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#based on https://github.com/pytorch/examples/blob/master/mnist/main.py

class ConvNet(nn.Module):
    def __init__(self, num_channel_in, increase_factor=7, kernerl_size=4):
        super(Net, self).__init__()
        num_channels_out_2 =  num_channel_in * increase_factor ** 2
        num_channels_out_1 = num_channel_in * increase_factor
        self.conv1 = nn.Conv2d(num_channel_in,
                               num_channels_out_1,
                               kernel_size=kernel_size)
        self.conv1 = nn.Conv2d(num_channels_out_1,
                               num_channels_out_2,
                               kernel_size=kernel_size)
        self.fc1 = nn.Linear(14*18*50, 500)
        self.fc2 = nn.Linear(500, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
