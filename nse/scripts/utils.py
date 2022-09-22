from tkinter import N
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

def Lpde(state_af, state_bf, dt):
    nx = state_bf.shape[1]
    ny = state_bf.shape[2]
    device = state_af.device

    u_bf = state_bf[..., :2]
    p_bf = state_bf[..., -1].reshape(-1, nx, ny, 1)
    u_af = state_af[..., :2]

    # ux, uy, u_lap = fftd2D(u_bf, device)
    # px, py, _ = fftd2D(p_bf, device)

    ux, uy = fdmd2D(u_bf, device)
    px, py = fdmd2D(p_bf, device)
    uxx, _ = fdmd2D(ux, device)
    _, uyy = fdmd2D(uy, device)

    u_lap = uxx + uyy
    p_grad = torch.cat((px, py), -1)
    L_state = (u_af - u_bf) / dt + u_bf[..., 0].reshape(-1, nx, ny, 1) * ux + \
              u_bf[..., 1].reshape(-1, nx, ny, 1) * uy - 0.001 * u_lap + p_grad

    loss = (L_state ** 2).mean()

    return L_state

def fdmd2D(u, device):
    bs = u.shape[0]
    nx = u.shape[-3]
    ny = u.shape[-2]
    dimu = u.shape[-1]
    dx = 2.2 / nx
    dy = 0.41 / ny
    ux = torch.zeros(bs, nx, ny, dimu).to(device)
    uy = torch.zeros(bs, nx, ny, dimu).to(device)
    for i in range(nx-1):
        ux[:, i] = (u[:, i+1] - u[:, i]) / dx
    ux[:, -1] = ux[:, -2]
    for j in range(ny-1):
        uy[:, :, j] = (u[:, :, j+1] - u[:, :, j]) / dy
    uy[:, :, -1] = uy[:, :, -2]

    return ux, uy

def fftd2D(u, device):
    nx = u.shape[-3]
    ny = u.shape[-2]
    dimu = u.shape[-1]
    u_h = torch.fft.fft2(u, dim=[1, 2]).reshape(-1, nx, ny, dimu)

    k_x = torch.arange(-nx//2, nx//2) * 2 * torch.pi / 2.2
    k_y = torch.arange(-ny//2, ny//2) * 2 * torch.pi / 0.41
    k_x = torch.fft.fftshift(k_x)
    k_y = torch.fft.fftshift(k_y)
    k_x = k_x.reshape(nx, 1).repeat(1, ny).reshape(1,nx,ny,1).to(device)
    k_y = k_y.reshape(1, ny).repeat(nx, 1).reshape(1,nx,ny,1).to(device)
    lap = -(k_x ** 2 + k_y ** 2)

    ux_h = 1j * k_x * u_h
    uy_h = 1j * k_y * u_h
    ulap_h = lap * u_h

    ux = torch.fft.ifft2(ux_h, dim=[1, 2])
    uy = torch.fft.ifft2(uy_h, dim=[1, 2])
    u_lap = torch.fft.ifft2(ulap_h, dim=[1, 2])

    ux = torch.real(ux).reshape(-1, nx, ny, dimu)
    uy = torch.real(uy).reshape(-1, nx, ny, dimu)
    u_lap = torch.real(u_lap).reshape(-1, nx, ny, dimu)

    return ux, uy, u_lap

# def AD2D(u, device):
#     nv = u.shape[-2]
#     dimu = u.shape[-1]

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
    def __init__(self, data_path, mode='grid'):
        self.mode = mode
        self.obs, self.Cd, self.Cl, self.ctr = torch.load(data_path)
        if self.mode == 'grid':
            self.obs = self.obs[..., 2:]
            self.nx, self.ny = self.obs.shape[-3], self.obs.shape[-2]
        elif self.mode == 'vertex':
            self.nv = self.obs.shape[-2]
        self.nt = self.ctr.shape[-1]
        self.N0 = self.ctr.shape[0]
        self.Ndata = self.N0 * self.nt

    def split(self, Ng, tg):
        self.Ng = Ng
        self.tg = tg
        self.dt = tg * 0.01
        self.obs = self.obs[::Ng, ::tg]
        self.Cd, self.Cl = self.Cd[::Ng, tg-1::tg], self.Cl[::Ng, tg-1::tg]
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
        if self.mode=='grid':
            obs_mean = self.obs.mean([0, 1, 2, 3])
            _obs_mean = obs_mean.reshape(1, 1, 1, 1, -1).repeat(self.N0, self.nt+1, self.nx, self.ny, 1)
            obs_var = torch.sqrt(((self.obs - _obs_mean)**2).mean([0, 1, 2, 3]))
        elif self.mode=='vertex':
            obs_mean = self.obs.mean([0, 1, 2])
            _obs_mean = obs_mean.reshape(1, 1, 1, -1).repeat(self.N0, self.nt+1, self.nv, 1)
            obs_var = torch.sqrt(((self.obs - _obs_mean)**2).mean([0, 1, 2]))
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
        if self.mode=='grid':
            return self.N0, self.nt, self.nx, self.ny
        elif self.mode=='vertex':
            return self.N0, self.nt, self.nv
    
    def trans2Dataset(self, batch_size):
        NSE_data = NSE_Dataset(self, self.mode)
        train_data, test_data = random_split(NSE_data, [int(0.8 * self.Ndata), int(0.2 * self.Ndata)])
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader
        

class NSE_Dataset(Dataset):
    def __init__(self, data, mode='grid'):
        if (mode == 'grid'):
            N0, nt, nx, ny = data.get_params()
            obs, Cd, Cl, ctr = data.get_data()
            self.Ndata = data.Ndata
            Cd = Cd.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
            Cl = Cl.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
            ctr = ctr.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
            input_data = obs[:, :-1].reshape(-1, nx, ny, 3)
            output_data = obs[:, 1:].reshape(-1, nx, ny, 3)     #- input_data
        elif (mode == 'vertex'):
            N0, nt, nv = data.get_params()
            obs, Cd, Cl, ctr = data.get_data()
            self.Ndata = data.Ndata
            Cd = Cd.reshape(N0, nt, 1, 1).repeat([1, 1, nv, 1]).reshape(-1, nv, 1)
            Cl = Cl.reshape(N0, nt, 1, 1).repeat([1, 1, nv, 1]).reshape(-1, nv, 1)
            ctr = ctr.reshape(N0, nt, 1, 1).repeat([1, 1, nv, 1]).reshape(-1, nv, 1)
            input_data = obs[:, :-1].reshape(-1, nv, 5)
            output_data = obs[:, 1:].reshape(-1, nv, 5) 

        self.ipt = torch.cat((input_data, ctr), dim=-1)
        self.opt = torch.cat((output_data, Cd, Cl), dim=-1)
        
    def __len__(self):
        return self.Ndata

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ipt[idx])
        y = torch.FloatTensor(self.opt[idx])
        return x, y


class PredLog():
    def __init__(self, mode, length):
        self.length = length
        self.mode = mode
        self.loss1 = AverageMeter()
        self.loss2 = AverageMeter()
        self.loss3 = AverageMeter()
        self.loss4 = AverageMeter()
        self.loss_pde = AverageMeter()
    
    def update(self, loss1, loss2, loss3, loss4, loss_pde):
        self.loss1.update(loss1.item(), self.length)
        self.loss2.update(loss2.item(), self.length)
        self.loss3.update(loss3.item(), self.length)
        self.loss4.update(loss4.item(), self.length)
        self.loss_pde.update(loss_pde.item(), self.length)

    def save_log(self, logs):
        if self.mode =='train':
            logs['train_loss_trans'].append(self.loss1.avg)
            logs['train_loss_u_t_rec'].append(self.loss2.avg)
            logs['train_loss_ctr_t_rec'].append(self.loss3.avg)
            logs['train_loss_trans_latent'].append(self.loss4.avg)
            logs['train_loss_pde'].append(self.loss_pde.avg)
        elif self.mode =='test':
            logs['test_loss_trans'].append(self.loss1.avg)
            logs['test_loss_u_t_rec'].append(self.loss2.avg)
            logs['test_loss_ctr_t_rec'].append(self.loss3.avg)
            logs['test_loss_trans_latent'].append(self.loss4.avg)
            logs['test_loss_pde'].append(self.loss_pde.avg)