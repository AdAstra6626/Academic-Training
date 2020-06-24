import torch
from torch import nn
import torch.nn.functional as F

class BaseLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([])
        '''  #base layer for imagenet 
        self.layers.append(nn.Conv2d(3,64,7,stride=2))
        if(cfg['bn']==True):
            self.layers.append(nn.BatchNorm2d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.MaxPool2d(3,stride=2))
        '''
        #base layer for cifar-10
        self.layers.append(nn.Conv2d(3,16,3,stride=1, padding=1))
        if(cfg['bn']==True):
            self.layers.append(nn.BatchNorm2d(16))
        self.layers.append(nn.ReLU(True))
    def forward(self,x):
        for ll in self.layers:
            x = ll(x)
        return x
        #cifar-10 shape: 32*32

#basic residual block
class BasicBlock(nn.Module):
    def __init__(self, in_channels=None, out_channels=None, block_type=None, stride=1, bn=True, 
     in_channels_1=None, in_channels_2=None, in_channels_3=None,
     out_channels_1=None, out_channels_2=None, out_channels_3=None):
        super().__init__()
        self.block_type = block_type
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.have_shortcut = 0
        self.bn = bn
        if self.block_type == 'small':
        #small resnet(18,34)
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=self.stride, padding=1)
            self.bn1 = nn.BatchNorm2d(self.out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(self.out_channels)
            self.rconv1 =  nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
            self.rconv2 =  nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1)
            self.rbn1 = nn.BatchNorm2d(self.out_channels)
            self.rbn2 = nn.BatchNorm2d(self.out_channels)
            if(self.stride != 1):
                self.have_shortcut = 1
                self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=self.stride)
                if(bn == 1):
                    self.shortcut_bn = nn.BatchNorm2d(self.out_channels)


    def forward(self, x):
        x0 = x
        x = self.conv1(x)
        x1_0 = F.relu(self.bn1(x))
        x1_1 = self.rconv1(x1_0)
        x1_1 += x 
        x1_1 = F.relu(self.rbn1(x))
        x2 = self.conv2(x1_1)
        x2_0 = F.relu(self.bn2(x))
        x2_1 = self.rconv2(x2_0)
        x2_1 += x2
        x2_1 = F.relu(self.rbn2(x2_1))
        if self.stride != 1 :
            x0 = self.shortcut_bn(self.shortcut(x0))
        x2_1 += x0
        return F.relu(x2_1)


class ResNet_recurrent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.type = cfg['type']
        self.bn = cfg['bn']
        self.in_channels = 64
        self.layer_info = {
            'ResNet_recurrent20':[3,3,3],
            'ResNet_recurrent32':[5,5,5],
            'ResNet_recurrent44':[7,7,7],
            'ResNet_recurrent110':[18,18,18],
        }
        self.BaseLayer = BaseLayer(cfg)
        self.block_type = 'small'
        self.ResBlocks = self.generate_layers(self.layer_info[self.type])
        if self.block_type == 'small':
            self.fc = nn.Linear(64, 10)
        else:
            self.fc = nn.Linear(2048,10)
  
    def generate_layers(self,_type):
        Blocks = nn.ModuleList([])
        if self.block_type == 'small':
            for i in range(_type[0]):
                Blocks.append(BasicBlock(16, 16, self.block_type, 1, self.bn))
            Blocks.append(BasicBlock(16, 32, self.block_type, 2, self.bn))
            for i in range(_type[1]-1):
                Blocks.append(BasicBlock(32, 32, self.block_type, 1, self.bn))
            Blocks.append(BasicBlock(32, 64, self.block_type, 2, self.bn))
            for i in range(_type[2]-1):
                Blocks.append(BasicBlock(64, 64, self.block_type, 1, self.bn))
        return Blocks
        
    
    def forward(self,x):
        x = self.BaseLayer.forward(x)
        for bb in self.ResBlocks:
            x = bb.forward(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

        
