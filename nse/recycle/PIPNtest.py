import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from  matplotlib import colors
import numpy as np

from scripts.utils import *

def f(x, y):
    # return torch.exp(x + y)
    return x + y

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

class test3(nn.Module):
    def __init__(self):
        super(test3, self).__init__()
        self.fc0 = nn.Conv2d(2, 16, 5, padding=2)
        self.fc1 = nn.Conv2d(16, 64, 5, padding=2)
        self.fc2 = nn.Conv2d(64, 4, 5, padding=2)
        self.fc3 = nn.Conv2d(4, 1, 5, padding=2)
    
    def forward(self, x):
        x = self.fc0(x)
        x = F.gelu(x)
        x = self.fc1(x)
        x = F.gelu(x)
        # print(xf.shape, x.shape)
        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        
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
xy = torch.cat((x, y), dim=-1).permute(2, 0, 1)
ipt = torch.cat((x, y), dim=-1).reshape(-1, 2)
z = f(x, y).permute(2, 0, 1)
opt = f(x, y).reshape(-1, 1)

data = testData(ipt, opt)
train_loader = DataLoader(dataset=data, batch_size=batchsize, shuffle=True)

def model_train(model, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(1, epochs + 1):
        model.train()
        
        for x, y in train_loader:
            
            optimizer.zero_grad()
            ym = model(x)
            loss = ((y - ym)**2).sum()

            loss.backward()
            optimizer.step()

        if(epoch% (epochs//10)==0): 
            print(f'# {epoch} loss: {loss}')
    model.eval()

model1 = test1()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-2)
model2 = test2()
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
# model_train(model1, epochs)
# model_train(model2, epochs)

model3 = test3()
optimizer3 = torch.optim.Adam(model3.parameters(), lr=1e-2)
for epoch in range(1, epochs + 1):
    model1.train()
    model2.train()
    model3.train()

    optimizer3.zero_grad()

    zm3 = model3(xy)

    loss3 = ((z - zm3)**2).sum() 
    loss = loss3
    loss.backward()
    optimizer1.step()
    optimizer2.step()
    optimizer3.step()

    if(epoch% (epochs//10)==0): 
        print(f'# {epoch} loss: {loss}')
model3.eval()

num = 100
xt = (torch.arange(num) / num)
yt = (torch.arange(num) / num)
# xt = torch.rand(num)
# yt = torch.rand(num)
xt, yt = torch.meshgrid(xt, yt)
xt = xt.unsqueeze(-1)
yt = yt.unsqueeze(-1)
ipt = torch.cat((xt, yt), dim=-1).reshape(-1, 2)
zt = f(ipt[:, 0], ipt[:, 1]).squeeze()
zt_ = f(xt, yt).permute(2, 0, 1)
ipt1 = torch.cat((xt, yt), dim=-1).reshape(-1, 2).requires_grad_(True)
ipt2 = torch.cat((xt, yt), dim=-1).reshape(-1, 2).requires_grad_(True)
ipt3 = torch.cat((xt, yt), dim=-1).permute(2, 0, 1).requires_grad_(True)
zm1 = model1(ipt1).squeeze()
zm2 = model2(ipt2).squeeze()
zm3 = model3(ipt3)
var1 = abs_error(zm1, zt)
var2 = abs_error(zm2, zt)
var3 = abs_error(zm3, zt_)
print(var1, var2, var3)
zm1[1].backward(retain_graph=True)
ipt1.grad
zm2[1].backward(retain_graph=True)
ipt2.grad
zm3[0, 2, 2].backward(retain_graph=True)
ipt3.grad

# xt1 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
# xt2 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
# yt1 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
# yt2 = torch.rand(num).requires_grad_(True).unsqueeze(-1)
# ipt1 = torch.cat((xt1, yt1), dim=-1)
# ipt2 = torch.cat((xt2, yt2), dim=-1)
# zt1 = f(xt1, yt1).squeeze()
# zt2 = f(xt2, yt2).squeeze()

# zm1 = model1(ipt1).squeeze()
# zm2 = model2(ipt2).squeeze()
# # ym = model1(xt)

# # print(xt[0].item(), ym[0].item(), xt.grad.squeeze())

# # weight = torch.ones(ym.shape)
# # dyx = torch.autograd.grad(ym, xt, weight, retain_graph=True, create_graph=True, only_inputs=True)

# # print(((yt - ym)**2).sum())
# # print(dyx)

# plt.figure(figsize=(15, 12))
# ax = plt.subplot2grid((1,1), (0, 0))

# x = torch.arange(10000) / 10000
# y = f(x)
# ax.plot(x.detach().numpy(), y.detach().numpy(), color = 'black')
# ax.plot(x.detach().numpy(), model1(x).squeeze().detach().numpy(), color = 'red')
# ax.plot(x.detach().numpy(), model2(x).squeeze().detach().numpy(), color = 'yellow')
# ax.grid(True, lw=0.4, ls="--", c=".50")
# l = 0
# dl = 1
# ax.set_xlim(l, l+dl)
# ax.set_ylim(l, l+dl)