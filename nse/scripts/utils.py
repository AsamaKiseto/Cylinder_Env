import torch
import operator
import numpy as np
import os, sys
from functools import reduce
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

def rel_error(x, _x):
    """
    <ARGS>
    x : torch.Tensor shape of (B, *)
    _x : torch.Tensor shape of (B, *)
    <RETURN>
    out :torch.Tensor shape of (B), batchwise relative error between x and _x : (||x-_x||_2/||_x||_2)
    
    """
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)
    
    return torch.norm(x - _x, 2, dim=1) / torch.norm(_x, 2, dim=1)


def abs_error(x, _x):
    """
    <ARGS>
    x : torch.Tensor shape of (B, *)
    _x : torch.Tensor shape of (B, *)
    <RETURN>
    out :torch.Tensor shape of (B), batchwise relative error between x and _x : (||x-_x||_2/||_x||_2)
    
    """
    if len(x.shape)==1:
        x = x.reshape(1, -1)
        _x = _x.reshape(1, -1)
    else:
        B = x.size(0)
        x, _x = x.reshape(B, -1), _x.reshape(B, -1)
    
    return torch.norm(x - _x, 2, dim=1) 

def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ReadData:
    def __init__(self, data_path):
        self.obs, _, self.Cd, self.Cl, self.ctr = torch.load(data_path)
        self.obs = self.obs[..., 2:]
        self.nx, self.ny = self.obs.shape[-3], self.obs.shape[-2]
        self.nt = self.ctr.shape[-1]
        self.N0 = self.ctr.shape[0]
        self.Ndata = self.N0 * self.nt

    def split(self, Ng, tg):
        self.obs = self.obs[::Ng, ::tg]
        self.Cd, self.Cl = self.Cd[::Ng, ::tg], self.Cl[::Ng, ::tg]
        self.ctr = self.ctr[::Ng, ::tg]
        self.get_params()
        return self.obs, self.Cd, self.Cl, self.ctr

    def norm(self):
        Cd_mean = self.Cd.mean()
        Cd_var = torch.sqrt(((self.Cd-Cd_mean)**2).mean())
        Cl_mean = self.Cl.mean()
        Cl_var = torch.sqrt(((self.Cl-Cl_mean)**2).mean())
        ctr_mean = self.ctr.mean()
        ctr_var = torch.sqrt(((self.ctr-ctr_mean)**2).mean())
        obs_mean = self.obs.mean([0, 1, 2, 3])
        _obs_mean = obs_mean.reshape(1, 1, 1, 1, -1).repeat(self.N0, self.nt+1, self.nx, self.ny, 1)
        obs_var = torch.sqrt(((self.obs - _obs_mean)**2).mean([0, 1, 2, 3]))

        self.Cd = (self.Cd - Cd_mean)/Cd_var
        self.Cl = (self.Cl - Cl_mean)/Cl_var

        self.norm = dict()
        self.norm['Cd'] = [Cd_mean, Cd_var]
        self.norm['Cl'] = [Cl_mean, Cl_var]
        self.norm['ctr'] = [ctr_mean, ctr_var]
        self.norm['obs'] = [obs_mean, obs_var]

        return self.norm

    def get_data(self):
        return self.obs, self.Cd, self.Cl, self.ctr

    def get_params(self):
        self.nt = self.ctr.shape[-1]
        self.N0 = self.ctr.shape[0]
        self.Ndata = self.N0 * self.nt
        return self.N0, self.nt, self.nx, self.ny
    
    def trans2Dataset(self):
        NSE_data = NSE_Dataset(self)
        train_data, test_data = random_split(NSE_data, [int(0.8 * self.Ndata), int(0.2 * self.Ndata)])
        return train_data, test_data
    
class NSE_Dataset(Dataset):
    def __init__(self, data):
        N0, nt, nx, ny = data.get_params()
        obs, Cd, Cl, ctr = data.get_data()
        self.Ndata = data.Ndata
        Cd = Cd.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
        Cl = Cl.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
        ctr = ctr.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
        input_data = obs[:, :-1].reshape(-1, nx, ny, 3)
        output_data = obs[:, 1:].reshape(-1, nx, ny, 3) #- input_data

        self.ipt = torch.cat((input_data, ctr), dim=-1)
        self.opt = torch.cat((output_data, Cd, Cl), dim=-1)
        
    def __len__(self):
        return self.Ndata

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ipt[idx])
        y = torch.FloatTensor(self.opt[idx])
        return x, y