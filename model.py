# Contains the neural network
# Task:- Implement the Neural Network module according to problem statement specifications


from torch import nn
from torch import reshape
import torch

''' 9-3-5 SRCNN model'''

class Net(nn.Module):
    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 9, padding=4)
        self.conv2 = nn.Conv2d(32,64,3, padding = 1)
        self.conv3 = nn.Conv2d(64, num_channels, 5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x