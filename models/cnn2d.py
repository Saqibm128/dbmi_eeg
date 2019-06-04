from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

#based on https://github.com/pytorch/examples/blob/master/mnist/main.py

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 4)
        self.conv1 = nn.Conv2d(20, 50, 4)
        self.fc1 = nn.Linear(14*18*50, 500)
        self.fc2 = nn.Linear(500, 8)

    def forward(self, x):
        x = F.relu(self.conv1(x))
