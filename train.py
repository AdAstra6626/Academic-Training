#%%
from torch.utils.data import DataLoader,Dataset
from load_data import MyDataset
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from resnet import ResNet

def Train(cfg):    
    mydataset = MyDataset()
    mydataloader = DataLoader(dataset=mydataset, batch_size = cfg["batch_size"], shuffle = True)
    dataiter = iter(mydataloader)
    device = torch.device('cuda:0')
    net = ResNet(cfg)
    epoch = cfg["epoches"]
    lr = cfg["lr"]
    momentum = cfg["momentum"]
    wd = cfg["weight_decay"]
    for param_tensor in net.state_dict():
        print(param_tensor,"\t",net.state_dict()[param_tensor].size())
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    

    #for i in range(epoch):

    

cfg = {"type":"ResNet18", "bn":True, "batch_size":32, "epoches":10, "lr":0.01, "momentum":0.9, "weight_decay":0.0005}
Train(cfg)

# %%
