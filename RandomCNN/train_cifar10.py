#%%

from torch.utils.data import DataLoader,Dataset
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from model import CNN, CNN_SE
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
import torchvision.transforms as transforms
from utils import progress_bar

def Train(cfg):

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, drop_last=True)  

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg["batch_size"], shuffle=False, num_workers=2, drop_last=True)
    device = torch.device(cfg["device"])
    if cfg["name"] == "CNN":
        net = CNN(cfg)
    elif cfg["name"] == "CNN_SE":
        net = CNN_SE(cfg)
    net.to(device)
    #net = nn.DataParallel(net, device_ids=[0,1,2,3])
    
    epoch = cfg["epoches"]
    lr = cfg["lr"]
    momentum = cfg["momentum"]
    wd = cfg["weight_decay"]

    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = MultiStepLR(optimizer, milestones=[80,160], gamma=0.1)
    acc_f = open('./result/{}_cifar10_acc.txt'.format(str(cfg["type"])),'w')
    loss_f = open('./result/{}_cifar10_loss.txt'.format(str(cfg["type"])),'w')
    acct_f = open('./result/{}_cifar10_acct.txt'.format(str(cfg["type"])),'w')
    acct_f.write(str(cfg))
    acct_f.write('\n')
    acct_f.flush()
    epoch_acc = []
    for i in range(epoch):
        net.train()
        acc_arr = []
        loss_arr = []
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for j, (data, label) in enumerate(trainloader):
            optimizer.zero_grad()
            out = net(data.float().to(device))
            loss = criterion(out, label.to(device))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += predicted.eq(label.data.to(device)).sum()
            progress_bar(j, len(trainloader), 'Epoch:%d/%d Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (i, epoch, sum_loss/(j+1), 100.*correct/total, correct, total))
            loss_arr.append(float(sum_loss / (j + 1)))
            acc_arr.append(float(100. * correct / total))
        acc_f.write(str(acc_arr))
        acc_f.write('\n')
        loss_f.write(str(loss_arr))
        loss_f.write('\n')
        acc_f.flush()
        loss_f.flush()
        
        with torch.no_grad():
            correct = 0
            total = 0
            for k,data in enumerate(testloader):
                net.eval()
                images, labels = data
                images, labels = images.float().to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
        progress_bar(k, len(testloader), 'Acc: %.3f%% (%d/%d)'
                         % (100.*correct/total, correct, total))
        acc = 100. * correct / total
        epoch_acc.append(acc)
        acct_f.write(str(acc))
        acct_f.write('\n')    
        acct_f.flush()
        scheduler.step()


cfg = {"name":"CNN", "type":"random_CNN_convert1", "bn":1, "batch_size":128, "epoches":240, "lr":0.1, "momentum":0.9, "weight_decay":0.0005, 
"device":"cuda:3", "nodes":32, 'graph_model':'convert', 'K':4, 'P' : 0.75, 'M':None, 'seed':1, "channels":78,
"reduction":16,
"rewire_K": 4,
"rewire_p":0.6}

if __name__ == "__main__":
    Train(cfg)


# %%
