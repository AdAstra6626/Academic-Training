#%%
import torch
from torch import nn
import torch.nn.functional as F

class rcnnlayer_shared(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride)
        self.rconv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

    
    def forward(self, x):
        ''' #iter3
        x = self.conv(x)
        x0 = F.relu(self.bn0(x))
        x1 = self.rconv(x0)
        x1 += x
        x1 = F.relu(self.bn1(x1))
        x2 = self.rconv(x1)
        x2 += x 
        x2 = F.relu(self.bn2(x2))
        x3 = self.rconv(x2)
        x3 += x
        x3 = F.relu(self.bn3(x3))
        return x3
        '''
        '''  #iter1
        x = self.conv(x)
        x0 = F.relu(self.bn0(x))
        x1 = self.rconv(x0)
        x1 += x
        x1 = F.relu(self.bn1(x1))
        return x1
        '''
          #iter2
        x = self.conv(x)
        x0 = F.relu(self.bn0(x))
        x1 = self.rconv(x0)
        x1 += x
        x1 = F.relu(self.bn1(x1))
        x2 = self.rconv(x1)
        x2 += x 
        x2 = F.relu(self.bn2(x2))
        return x2
        
        '''    #base
        x = self.conv(x)
        x0 = F.relu(self.bn0(x))
        return x0
        '''

class rcnnlayer_unshared(nn.Module):
    def __init__(self, in_channels, stride):
        super().__init__()
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride)
        self.rconv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride)
        self.rconv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride)
        self.rconv3 = nn.Conv2d(in_channels, in_channels, 3, padding=1, stride=stride)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.bn3 = nn.BatchNorm2d(in_channels)

    
    def forward(self, x):
        '''
        x = self.conv(x)
        x0 = F.relu(self.bn0(x))
        x1 = self.rconv1(x0)
        x1 += x
        x1 = F.relu(self.bn1(x1))
        x2 = self.rconv2(x1)
        x2 += x 
        x2 = F.relu(self.bn2(x2))
        x3 = self.rconv3(x2)
        x3 += x
        x3 = F.relu(self.bn3(x3))
        return x3
        '''
        x = self.conv(x)
        x0 = F.relu(self.bn0(x))
        x1 = self.rconv1(x0)
        x1 += x
        x1 = F.relu(self.bn1(x1))
        x2 = self.rconv2(x1)
        x2 += x 
        x2 = F.relu(self.bn2(x2))
        return x2

class rcnn1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 192, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(192)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.rconv1 = rcnnlayer_shared(192, 1)
        self.rconv2 = rcnnlayer_shared(192, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.rconv3 = rcnnlayer_shared(192, 1)
        self.rconv4 = rcnnlayer_shared(192, 1)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.rconv1(x)
        x = self.rconv2(x)
        x = self.pool2(x)
        x = self.rconv3(x)
        x = self.rconv4(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class rcnn_unshared(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 192, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(192)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.rconv1 = rcnnlayer_unshared(self.batch_size, 192, 1)
        self.rconv2 = rcnnlayer_unshared(self.batch_size, 192, 1)
        self.pool2 = nn.MaxPool2d(2)
        self.rconv3 = rcnnlayer_unshared(self.batch_size, 192, 1)
        self.rconv4 = rcnnlayer_unshared(self.batch_size, 192, 1)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.rconv1(x)
        x = self.rconv2(x)
        x = self.pool2(x)
        x = self.rconv3(x)
        x = self.rconv4(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
with torch.cuda.device(0):
  net = rcnn1()
  macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))
# %%
