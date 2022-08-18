import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt 
from  matplotlib import colors
import numpy as np

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

def f(x):
    return x

class test1(nn.Module):
    def __init__(self):
        super(test1, self).__init__()
        self.width1 = 4
        self.width2 = 8
        self.fc0 = nn.Linear(1, self.width1)
        self.fc1 = nn.Linear(self.width1, self.width2)
        self.fc2 = nn.Linear(self.width1+self.width2 , self.width1)
        self.fc3 = nn.Linear(self.width1, 1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1)
        x = self.fc0(x)
        # x = F.gelu(x)
        xf = self.fc1(x)
        # xf = F.gelu(xf)
        # print(xf.shape, x.shape)
        xf = xf.mean(0).reshape(1, self.width2).repeat(x.shape[0], 1)
        x = torch.cat((x, xf), dim=-1)
        x = self.fc2(x)
        # x = F.gelu(x)
        x = self.fc3(x)
        
        return x

class test2(nn.Module):
    def __init__(self):
        super(test2, self).__init__()
        self.width1 = 4
        self.width2 = 8
        self.fc0 = nn.Linear(1, self.width1)
        self.fc1 = nn.Linear(self.width1, self.width2)
        self.fc2 = nn.Linear(self.width2, self.width1)
        self.fc3 = nn.Linear(self.width1, 1)
    
    def forward(self, x):
        x = x.reshape(x.shape[0], 1)
        x = self.fc0(x)
        # x = F.gelu(x)
        x = self.fc1(x)
        # x = F.gelu(x)
        # print(xf.shape, x.shape)
        x = self.fc2(x)
        # x = F.gelu(x)
        x = self.fc3(x)
        
        return x

class testData(Dataset):
    def __init__(self, x, y):
        self.ipt = x
        self.opt = y

    def __len__(self):
        return len(x)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ipt[idx])
        y = torch.FloatTensor(self.opt[idx])
        return x, y

N = 1000
epochs = 1000
num = 10

x = torch.rand(N)
x = torch.arange(N) / N
x = x.unsqueeze(-1)
y = f(x)

data = testData(x, y)
train_loader = DataLoader(dataset=data, batch_size=N, shuffle=True)

model1 = test1()
model2 = test2()
optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-2)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-2)

for epoch in range(1, epochs + 1):
    model1.train()
    
    for x, y in train_loader:
        
        optimizer1.zero_grad()
        ym = model1(x)
        loss = ((y - ym)**2).mean()

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
        loss = ((y - ym)**2).mean()

        loss.backward()
        optimizer2.step()

        if(epoch%100==0): 
            print(f'# {epoch} loss: {loss}')

model2.eval()

# xt1 = torch.tensor([0.5]).requires_grad_(True)
# xt2 = torch.tensor([0.5]).requires_grad_(True)
xt1 = torch.rand(num).requires_grad_(True)
xt2 = torch.rand(num).requires_grad_(True)
yt1 = f(xt1)
yt2 = f(xt2)
xt = x.requires_grad_(True)
ym1 = model1(xt1).squeeze()
ym2 = model2(xt2).squeeze()
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