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
        self.layers.append(nn.Conv2d(3,64,3,stride=1, padding=1).cuda())
        if(cfg['bn']==True):
            self.layers.append(nn.BatchNorm2d(64).cuda())
        self.layers.append(nn.ReLU().cuda())
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
        self.layers = nn.ModuleList([])
        self.have_bottleneck = 0
        self.bn = bn
        if self.block_type == 'small':
        #small resnet(18,34)
            self.layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=self.stride, padding=1).cuda())
            if(bn == 1):
                self.layers.append(nn.BatchNorm2d(self.out_channels).cuda())
            self.layers.append(nn.ReLU().cuda())
            self.layers.append(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1).cuda())
            if(bn == 1):
                self.layers.append(nn.BatchNorm2d(self.out_channels).cuda())
            self.layers.append(nn.ReLU().cuda())
            if(self.stride != 1):
                self.have_bottleneck = 1
                self.bottleneck = nn.Conv2d(in_channels, out_channels, 1, stride=self.stride).cuda()
                if(bn == 1):
                    self.bottleneck_bn = nn.BatchNorm2d(self.out_channels).cuda()
        else:
        #big resnet(50, 101, 152)
            self.layers.append(nn.Conv2d(in_channels_1, out_channels_1, 1, stride=self.stride).cuda())
            if(bn == 1):
                self.layers.append(nn.BatchNorm2d(self.out_channels_1).cuda())
            self.layers.append(nn.ReLU().cuda())
            self.layers.append(nn.Conv2d(in_channels_2, out_channels_2, 3, stride=1, padding=1).cuda())
            if(bn == 1):
                self.layers.append(nn.BatchNorm2d(self.out_channels_2).cuda())
            self.layers.append(nn.ReLU().cuda())
            self.layers.append(nn.Conv2d(in_channels_3, out_channels_3, 1, stride=1).cuda())
            if(bn == 1):
                self.layers.append(nn.BatchNorm2d(self.out_channels_3).cuda())
            self.layers.append(nn.ReLU().cuda())
            self.have_bottleneck = 1
            self.bottleneck = nn.Conv2d(in_channels_1, out_channels_3, 1, stride=self.stride).cuda()
            if(bn == 1):
                self.bottleneck_bn = nn.BatchNorm2d(self.out_channels_3).cuda()


    def forward(self, x):
        out = x
        for ll in self.layers:
            out = ll(out)
        if self.have_bottleneck != 0:
            x = self.bottleneck(x)
            if self.bn == 1:
                x = self.bottleneck_bn(x)
            out += x
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.type = cfg['type']
        self.bn = cfg['bn']
        self.in_channels = 64
        self.layer_info = {
            'ResNet18':[2,2,2,2],
            'ResNet34':[3,4,6,3],
            'ResNet50':[3,4,6,3],
            'ResNet101':[3,4,23,3],
            'ResNet152':[3,8,36,3]
        }
        self.BaseLayer = BaseLayer(cfg)
        if self.type == 'ResNet18' or self.type == 'ResNet34':
            self.block_type = 'small'
        else:
            self.block_type = 'big'
        self.ResBlocks = self.generate_layers(self.layer_info[self.type])
        if self.block_type == 'small':
            self.fc = nn.Linear(512, 10).cuda()
        else:
            self.fc = nn.Linear(2048,10).cuda()

    
    def generate_layers(self,_type):
        Blocks = nn.ModuleList([])
        if self.block_type == 'small':
            for i in range(_type[0]):
                Blocks.append(BasicBlock(64, 64, self.block_type, 1, self.bn))
            Blocks.append(BasicBlock(64, 128, self.block_type, 2, self.bn))
            for i in range(_type[1]-1):
                Blocks.append(BasicBlock(128, 128, self.block_type, 1, self.bn))
            Blocks.append(BasicBlock(128, 256, self.block_type, 2, self.bn))
            for i in range(_type[2]-1):
                Blocks.append(BasicBlock(256, 512, self.block_type, 1, self.bn))
            for i in range(_type[3]-1):
                Blocks.append(BasicBlock(512, 512, self.block_type, 1, self.bn))

        else:
            Blocks.append(BasicBlock(block_type = self.block_type, stride=1, bn=self.bn,
                in_channels_1=64, in_channels_2=64, in_channels_3=64,
                out_channels_1=64, out_channels_2=64, out_channels_3=256))
            for i in range(_type[0]-1):
                Blocks.append(BasicBlock(block_type = self.block_type, stride=1, bn=self.bn,
                in_channels_1=256, in_channels_2=64, in_channels_3=64,
                out_channels_1=64, out_channels_2=64, out_channels_3=256))

            Blocks.append(BasicBlock(block_type = self.block_type, stride=2, bn=self.bn,
                in_channels_1=256, in_channels_2=128, in_channels_3=128,
                out_channels_1=128, out_channels_2=128, out_channels_3=512))
            for i in range(_type[1]-1):
                Blocks.append(BasicBlock(block_type = self.block_type, stride=1, bn=self.bn,
                in_channels_1=512, in_channels_2=128, in_channels_3=128,
                out_channels_1=128, out_channels_2=128, out_channels_3=512))

            Blocks.append(BasicBlock(block_type = self.block_type, stride=2, bn=self.bn,
                in_channels_1=512, in_channels_2=256, in_channels_3=256,
                out_channels_1=256, out_channels_2=256, out_channels_3=1024))
            for i in range(_type[2]-1):
                Blocks.append(BasicBlock(block_type = self.block_type, stride=1, bn=self.bn,
                in_channels_1=1024, in_channels_2=256, in_channels_3=256,
                out_channels_1=256, out_channels_2=256, out_channels_3=1024))

            Blocks.append(BasicBlock(block_type = self.block_type, stride=2, bn=self.bn,
                in_channels_1=1024, in_channels_2=512, in_channels_3=512,
                out_channels_1=512, out_channels_2=512, out_channels_3=2048))
            for i in range(_type[3]-1):
                Blocks.append(BasicBlock(block_type = self.block_type, stride=1, bn=self.bn,
                in_channels_1=2048, in_channels_2=512, in_channels_3=512,
                out_channels_1=512, out_channels_2=512, out_channels_3=2048))        
        return Blocks
        
    
    def forward(self,x):
        x = self.BaseLayer.forward(x)
        for bb in self.ResBlocks:
            x = bb.forward(x)
        x = F.avg_pool2d(x, x.size()[-1])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


        
