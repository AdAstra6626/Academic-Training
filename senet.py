#%%
import torch
from torch import nn
import torch.nn.functional as F

#%%
class semodule(nn.Module):
    def __init__(self, r, in_channels, mid_channels, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)
        self.W1 = nn.Linear(self.mid_channels, int(self.mid_channels/r), bias=False)
        self.W2 = nn.Linear(int(self.mid_channels/r), self.mid_channels, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, stride=stride)
        self.batch_norm2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(self.mid_channels, self.out_channels, 1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    

    def forward(self, x):
        x0 = x
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = F.relu(self.batch_norm1(x))
        u = F.relu(self.batch_norm2(self.conv2(x)))
        u0 = u
        h_w = u.size()[-1]
        u = F.avg_pool2d(u, u.size()[-1])
        u = u.view(-1,self.mid_channels)
        u = F.sigmoid(self.W2(F.relu(self.W1(u))))
        u = torch.unsqueeze(u,2)  
        u = u.expand(batch_size, self.mid_channels, h_w*h_w)
        u = u.resize(batch_size, self.mid_channels, h_w, h_w)
        u0 = torch.mul(u, u0)
        out = self.conv3(u0)
        x0 = self.shortcut(x0)
        out += x0
        return F.relu(out)


class senet(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.layers = nn.ModuleList([])
        self.layers.append(semodule(16, 64, 128, 256, 1))
        for i in range(2):
            self.layers.append(semodule(16, 256, 128, 256, 1))
        self.layers.append(semodule(16, 256, 256, 512, 2))
        for i in range(2):
            self.layers.append(semodule(16, 512, 256, 512, 1))
        self.layers.append(semodule(16, 512, 512, 1024, 2))
        for i in range(2):
            self.layers.append(semodule(16, 1024, 512, 1024, 1))

        self.fc = nn.Linear(1024, 10)
        '''
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(32)

        self.layers = nn.ModuleList([])
        self.layers.append(semodule(16, 32, 64, 128, 1))
        for i in range(2):
            self.layers.append(semodule(16, 128, 64, 128, 1))
        self.layers.append(semodule(16, 128, 128, 256, 2))
        for i in range(2):
            self.layers.append(semodule(16, 256, 128, 256, 1))
        self.layers.append(semodule(16, 256, 256, 512, 2))
        for i in range(2):
            self.layers.append(semodule(16, 512, 256, 512, 1))

        self.fc = nn.Linear(512, 10)
    
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
  net = senet()
  macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
  print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  print('{:<30}  {:<8}'.format('Number of parameters: ', params))

# %%
