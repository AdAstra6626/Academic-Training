#%%
import torch
from torch import nn
import torch.nn.functional as F


class skmodule(nn.Module):
    def __init__(self, G, r, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.conv2_1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.W = nn.Linear(self.mid_channels, int(self.mid_channels/r), bias=False)
        self.batch_norm_w = nn.BatchNorm1d(int(self.mid_channels/r))
        self.conv2_2_1 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, groups=G, stride=stride)
        self.batch_norm2_1 = nn.BatchNorm2d(mid_channels)
        #self.conv2_2_2 = nn.Conv2d(mid_channels, mid_channels, 3, dilation = 2, padding=2, groups=G, stride=stride)
        self.conv2_2_2 = nn.Conv2d(mid_channels, mid_channels, 1, stride=stride, groups=G)
        self.batch_norm2_2 = nn.BatchNorm2d(mid_channels)
        self.A = nn.Linear(int(self.mid_channels/r), self.mid_channels, bias=False)
        self.conv3 = nn.Conv2d(self.mid_channels, self.out_channels, 1)
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    

    def forward(self, x):
        x0 = x
        batch_size = x.size()[0]
        x = self.conv2_1(x)
        x = F.relu(self.batch_norm1(x))
        u1 = F.relu(self.batch_norm2_1(self.conv2_2_1(x)))
        u2 = F.relu(self.batch_norm2_2(self.conv2_2_2(x)))
        u = u1 + u2
        u = F.avg_pool2d(u, u.size()[-1])
        u = u.view(-1,self.mid_channels)
        u = F.relu(self.batch_norm_w(self.W(u)))
        a = self.A(u)
        h_w = u1.size()[-1]
        ac = torch.exp(a)/(torch.exp(a)+ torch.exp(b)) 
        ac = torch.unsqueeze(ac,2)  
        ac = ac.expand(batch_size, self.mid_channels, h_w*h_w)
        ac = ac.resize(batch_size, self.mid_channels, h_w, h_w)
        bc = 1 - ac
        u1 = torch.mul(u1, ac)
        u2 = torch.mul(u2, bc)
        out = u1 + u2
        out = self.conv3(out)
        if self.in_channels != self.out_channels:
            x0 = self.shortcut(x0)
        out += x0
        return F.relu(out)


class sknet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.layers = nn.ModuleList([])
        #base on 16*32d
        self.layers.append(skmodule(16, 16, 64, 512, 256, 1))
        for i in range(2):
            self.layers.append(skmodule(16, 16, 256, 512, 256, 1))
        self.layers.append(skmodule(16, 16, 256, 1024, 512, 2))
        for i in range(2):
            self.layers.append(skmodule(16, 16, 512, 1024, 512, 1))
        self.layers.append(skmodule(16, 16, 512, 2048, 1024, 2))
        for i in range(2):
            self.layers.append(skmodule(16, 16, 1024, 2048, 1024, 1))

        self.fc = nn.Linear(1024, 10)


        '''
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.layers = nn.ModuleList([])
        self.layers.append(skmodule(32, 16, 32, 64, 128, 1))
        for i in range(2):
            self.layers.append(skmodule(32, 16, 128, 64, 128, 1))
        self.layers.append(skmodule(32, 16, 128, 128, 256, 2))
        for i in range(2):
            self.layers.append(skmodule(32, 16, 256, 128, 256, 1))
        self.layers.append(skmodule(32, 16, 256, 256, 512, 2))
        for i in range(2):
            self.layers.append(skmodule(32, 16, 512, 256, 512, 1))

        self.fc = nn.Linear(512, 10)
        '''
    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        for n in self.layers:
            x = n(x)
        out = F.avg_pool2d(x, 8)
        out = out.view(x.size()[0], -1)
        return self.fc(out)


import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

with torch.cuda.device(0):
  net = sknet()
  macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# %%
