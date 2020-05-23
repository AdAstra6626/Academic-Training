#%%
import torch
from torch import nn
import numpy as np 
import torch.nn.functional as F 

class VGG16(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.bn = cfg['bn']
        self.layers = nn.ModuleList([])
        self.classifier = nn.ModuleList([])
        self.add_conv_layer(3,64,self.bn)
        self.add_conv_layer(64,64,self.bn)
        self.layers.append(nn.MaxPool2d(2,2))
        self.add_conv_layer(64,128,self.bn)
        self.add_conv_layer(128,128,self.bn)
        self.layers.append(nn.MaxPool2d(2,2))
        self.add_conv_layer(128,256,self.bn)
        self.add_conv_layer(256,256,self.bn)
        self.add_conv_layer(256,256,self.bn)
        self.layers.append(nn.MaxPool2d(2,2))
        self.add_conv_layer(256,512,self.bn)
        self.add_conv_layer(512,512,self.bn)
        
        self.add_conv_layer(512,512,self.bn)
        self.layers.append(nn.MaxPool2d(2,2))
        self.add_conv_layer(512,512,self.bn)
        self.add_conv_layer(512,512,self.bn)
        self.add_conv_layer(512,512,self.bn)
        self.layers.append(nn.MaxPool2d(2,2))
        '''imagenet
        self.classifier.append(nn.Dropout())
        self.classifier.append(nn.Linear(25088,4096))
        self.classifier.append(nn.ReLU())
        self.classifier.append(nn.Dropout())
        self.classifier.append(nn.Linear(4096,4096))
        self.classifier.append(nn.ReLU())
        self.classifier.append(nn.Linear(4096,1000))
        '''
        self.classifier.append(nn.Linear(512,10))


    def add_conv_layer(self, inchannel, outchannel, bn):
        self.layers.append(nn.Conv2d(inchannel, outchannel,3,padding=1))
        if bn:
            self.layers.append(nn.BatchNorm2d(outchannel))
        self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x=layer(x)
        x = x.view(x.size(0), -1)
        for layer in self.classifier:
            x=layer(x)
        return x

