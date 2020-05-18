#%%
import torch
from torch import nn

# %%
class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
        for i in range(5):
            self.layers.append(nn.Linear(10,10).cuda())

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

testnet = TestNet()
data = torch.Tensor([1,2,3,4,5,6,7,8,9,0]).cuda()
testnet(data)

# %%


# %%
