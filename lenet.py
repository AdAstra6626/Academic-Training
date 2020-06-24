import torch
from torch import nn
import numpy as np 
import torch.nn.functional as F 

class lenet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Conv2d(1,6,5,padding=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Conv2d(6,16,5,padding=2))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(2))
        self.layers.append(nn.Conv2d(16,120,4,padding=2))
        self.layers.append(nn.ReLU())
        self.fc1 = nn.Linear(7680,64)
        self.fc2 = nn.Linear(64,10)

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x