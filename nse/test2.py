import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from  matplotlib import colors
import numpy as np

from scripts.utils import *

def f(x, y):
    return torch.exp(x + y)
    # return x + y

class test1(nn.Module):
    def __init__(self):
        super(test1, self).__init__()
        self.fc0 = nn.Linear(2, 8)
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(40, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc0(x)
        x = F.gelu(x)
        xf = self.fc1(x)
        xf = F.gelu(xf)
        # print(xf.shape, x.shape)
        xf = xf.mean(0).reshape(1, 32).repeat(x.shape[0], 1)
        x = torch.cat((x, xf), dim=-1)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        
        return x

class test2(nn.Module):
    def __init__(self):
        super(test2, self).__init__()
        self.fc0 = nn.Linear(2, 8)
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(4, 1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.fc0(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = F.gelu(x)
        # print(xf.shape, x.shape)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        
        return x

class testData(Dataset):
    def __init__(self, ipt, opt):
        self.ipt = ipt
        self.opt = opt

    def __len__(self):
        return self.ipt.shape[0]
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ipt[idx])
        y = torch.FloatTensor(self.opt[idx])
        return x, y

N = 100
epochs = 1000
batchsize = 1000
num = 10

x = torch.rand(N)
x = torch.arange(N) / N
y = x
x, y = torch.meshgrid(x, y)
x = x.unsqueeze(-1)
y = y.unsqueeze(-1)
ipt = torch.cat((x, y), dim=-1).reshape(-1, 2)
opt = f(x, y).reshape(-1, 1)

data = testData(ipt, opt)
train_loader = DataLoader(dataset=data, batch_size=batchsize, shuffle=True)

model1 = test1()
model2 = test2()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-2)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)

for epoch in range(1, epochs + 1):
    model1.train()
    
    for x, y in train_loader:
        
        optimizer1.zero_grad()
        ym = model1(x)
        loss = ((y - ym)**2).sum()

        loss.backward()
        optimizer1.step()

    if(epoch% (epochs//10)==0): 
        print(f'# {epoch} loss: {loss}')

model1.eval()

for epoch in range(1, epochs + 1):
    model2.train()
    
    for x, y in train_loader:
        
        optimizer2.zero_grad()
        ym = model2(x)
        loss = ((y - ym)**2).sum()

        loss.backward()
        optimizer2.step()

    if(epoch%100==0): 
        print(f'# {epoch} loss: {loss}')    

model2.eval()

# xt1 = torch.tensor([0.5]).requires_grad_(True)
# xt2 = torch.tensor([0.5]).requires_grad_(True)
num = 100

xt = (torch.arange(num) / num)
yt = (torch.arange(num) / num)
xt = torch.rand(num)
yt = torch.rand(num)
xt, yt = torch.meshgrid(xt, yt)
xt = xt.unsqueeze(-1)
yt = yt.unsqueeze(-1)
ipt = torch.cat((xt, yt), dim=-1).reshape(-1, 2)
zt = f(ipt[:, 0], ipt[:, 1]).squeeze()
ipt1 = ipt.requires_grad_(True)
ipt2 = ipt.requires_grad_(True)
zm1 = model1(ipt1).squeeze()
zm2 = model2(ipt2).squeeze()
var1 = abs_error(zm1, zt)
var2 = abs_error(zm2, zt)
print(var1, var2)
zm1[0].backward()
ipt1.grad


xt1 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
xt2 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
yt1 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
yt2 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
ipt1 = torch.cat((xt1, yt1), dim=-1)
ipt2 = torch.cat((xt2, yt2), dim=-1)
zt1 = f(xt1, yt1).squeeze()
zt2 = f(xt2, yt2).squeeze()

zm1 = model1(ipt1).squeeze()
zm2 = model2(ipt2).squeeze()
# ym = model1(xt)

k = 0
ym1[k].backward()
ym2[k].backward()
# ym[0].backward()
print(xt1[k].item(), ym1[k].item(), xt1.grad) 
print(xt2[k].item(), ym2[k].item(), xt2.grad)
# print(xt[0].item(), ym[0].item(), xt.grad.squeeze())

# weight = torch.ones(ym.shape)
# dyx = torch.autograd.grad(ym, xt, weight, retain_graph=True, create_graph=True, only_inputs=True)

# print(((yt - ym)**2).sum())
# print(dyx)

plt.figure(figsize=(15, 12))
ax = plt.subplot2grid((1,1), (0, 0))

x = torch.arange(10000) / 10000
y = f(x)
ax.plot(x.detach().numpy(), y.detach().numpy(), color = 'black')
ax.plot(x.detach().numpy(), model1(x).squeeze().detach().numpy(), color = 'red')
ax.plot(x.detach().numpy(), model2(x).squeeze().detach().numpy(), color = 'yellow')
ax.grid(True, lw=0.4, ls="--", c=".50")
l = 0
dl = 1
ax.set_xlim(l, l+dl)
ax.set_ylim(l, l+dl)