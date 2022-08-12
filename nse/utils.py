import torch
import operator
import numpy as np
import os, sys
from functools import reduce

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


def get_mask(is_augmented=False):    
    y_sample = np.load('./data/stokes_tri_256_u.npy')[0, 0]
    nonzero_mask = np.where(y_sample!=0)
    zero_mask = np.where(y_sample==0)
    
    if is_augmented:
        aug_zero = []
        for i,j in zip(*zero_mask):
            aug_zero += [(i,j-1), (i,j+1), (i-1,j), (i+1,j)]
        aug_zero = set(aug_zero)

        aug_nonzero = set([(i,j) for i,j in zip(*nonzero_mask)])
        aug_nonzero -= aug_zero

        aug_nonzero_mask = (np.array([i for i,_ in aug_nonzero]), 
                            np.array([j for _,j in aug_nonzero]))
        aug_zero_mask = (np.array([i for i,_ in aug_zero]), 
                         np.array([j for _,j in aug_zero]))
        
        return aug_nonzero_mask, aug_zero_mask
    else:
        return nonzero_mask, zero_mask
    
def coef2tensor(s):
    gs = 256
    x = torch.linspace(0, 30, gs)
    y = torch.linspace(0, 10, gs//4)
    mx, my = torch.meshgrid((x, y))
    grids = torch.stack((mx, my), 2)
    
    def get_torch_values(function, grids):
        out = torch.zeros(grids.shape[:2])
        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):
                xy = grids[i, j]
                try:
                    out[i, j] = function(xy)
                except RuntimeError:
                    out[i, j] = 0

        return out
    
    vel_1, vel_2 = get_torch_values(s.sub(0)[0], grids), get_torch_values(s.sub(0)[1], grids)
    pressure = get_torch_values(s.sub(1), grids)
    
    vel_1 = vel_1.permute(1,0)
    vel_2 = vel_2.permute(1,0)
    pressure = pressure.permute(1,0)
    
    return vel_1, vel_2, pressure
    
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size()))
    return c

