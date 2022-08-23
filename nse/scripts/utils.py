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

def nse_L(state):
    nx = state.shape[1]
    ny = state.shape[2]
    device = state.device

    u = state[..., :2]
    p = state[..., -1]
    u_h = torch.fft.fft2(u, dim=[1, 2])
    p_h = torch.fft.fft2(p, dim=[1, 2])
    print(u_h.shape)

    k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1, device=device),
                     torch.arange(start=-nx//2, end=0, step=1, device=device)), 0).reshape(nx, 1).repeat(1, ny).reshape(1,nx,ny,1)
    k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1, device=device),
                     torch.arange(start=-ny//2, end=0, step=1, device=device)), 0).reshape(1, ny).repeat(nx, 1).reshape(1,nx,ny,1)
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0

    ux_h = 1j * k_x * u_h       
    uy_h = 1j * k_y * u_h

    print(ux_h.shape) 
    px_h = 1j * k_x * p_h 
    py_h = 1j * k_x * p_h
    ulap_h = -lap * u_h
    plap_h = -lap * p_h

    ux = torch.fft.ifft2(ux_h, dim=[1, 2])
    uy = torch.fft.ifft2(uy_h, dim=[1, 2])
    px = torch.fft.ifft2(px_h, dim=[1, 2])
    py = torch.fft.ifft2(py_h, dim=[1, 2])
    u_lap = torch.fft.ifft2(ulap_h, dim=[1, 2])
    p_grad = torch.cat((px, py), -1)

    L_state = u[..., 0] * ux + u[..., 1] * uy - 0.0001 * u_lap + p_grad

    return L_state
    

def grad_fft(f):
    fhat = torch.fft.fft2(f, )



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
        self.obs, self.Cd, self.Cl, self.ctr = torch.load(data_path)
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

    def disnorm(self, norm):
        Cd_mean, Cd_var = norm['Cd']
        Cl_mean, Cl_var = norm['Cl']
        ctr_mean, ctr_var = norm['ctr']
        obs_mean, obs_var = norm['obs']

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

