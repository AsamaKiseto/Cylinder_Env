import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

N = 1000
x = torch.arange(N) / N
y = torch.arange(N) / N
x, y = torch.meshgrid(x, y)
x = x.unsqueeze(-1)
y = y.unsqueeze(-1)

x.requires_grad_(True)
y.requires_grad_(True)

z = torch.cat((x, y, x*y), dim=-1)

idx, idy, idz = 1, 1, 2
z[idx, idy, idz].backward()

print(x[idx, idy].item(), y[idx, idy].item(), z[idx, idy, idz], x.grad.squeeze()) 


