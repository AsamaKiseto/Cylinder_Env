import sys
sys.path.append("..")
sys.path.append("../env")

import numpy as np
from os import path
import math

import torch
from timeit import default_timer
import math
import torch.nn as nn
from env.RBC_env import RBC

params = {'dt':  0.05, 'T':  0.01, 'dimx': 64, 'dimy': 32, 'min_x' : 0, 'max_x' : 2.0, 'min_y' : 0.0, 'max_y' : 1.0 ,'Ra':1E6}
simulator = RBC(params)

Nf = 20 + 1
Nf = 5
N0 = Nf ** 2
nx = simulator.params['dimx']
ny = simulator.params['dimy']
dt = simulator.params['dt']
# end_t = 2.0
range_t = 1.0
nlt = int(range_t // dt) + 1
nt = 2 * nlt + 1
print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}, nlt: {nlt}')

temp, velo, p = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny))
# ctr1 = np.linspace(1.0, 3.0, Nf)
# ctr2 = np.linspace(1.0, 3.0, Nf)
ctr1 = np.random.rand(Nf) * 2 + 1
ctr2 = np.random.rand(Nf) * 2 + 1
ctr1_ = ctr1.reshape(Nf, 1).repeat(Nf, 1).reshape(N0, 1).repeat(nlt, 1)
ctr2_ = ctr2.reshape(1, Nf).repeat(Nf, 0).reshape(N0, 1).repeat(nlt, 1)
ctr = np.concatenate((ctr1_, ctr2_), -1)
print(ctr1, ctr2)
# print(ctr.shape, ctr)

for k in range(Nf):
    for l in range(Nf):
        print(f'start # {Nf * k + l + 1}')
        start = default_timer()

        simulator.reset(ctr=1.0, const=0.0)
        for i in range(int(nlt)):
            simulator.step()
        
        temp[Nf * k + l, 0], velo[Nf * k + l, 0], p[Nf * k + l, 0], _  = simulator.step()
    
        simulator.set(ctr=ctr1[k], const=0.0)
        for i in range(nlt):
            temp[Nf * k + l, i + 1], velo[Nf * k + l, i + 1], p[Nf * k + l, i + 1], _  = simulator.step()
        
        simulator.set(ctr=ctr2[l], const=0.0)
        for i in range(nlt):
            temp[Nf * k + l, i + nlt + 1], velo[Nf * k + l, i + nlt + 1], p[Nf * k + l, i + nlt + 1], _  = simulator.step()
    
        end = default_timer()

        print(f'end # {Nf * k + l + 1} | time: {end-start}')

temp = torch.Tensor(temp).reshape(N0, nt, nx, ny, 1)
velo = torch.Tensor(velo)
p = torch.Tensor(p).reshape(N0, nt, nx, ny, 1)
ctr = torch.Tensor(ctr)
obs = torch.cat((velo, p), dim=-1)

print(ctr.shape, obs.shape)

# torch.save([obs, temp, ctr], 'data/nse_data_reg_rbc_orig_test')
# torch.save([obs, temp], 'data/nse_data_reg_rbc_test')

# torch.save([obs, temp, ctr], 'data/nse_data_reg_rbc7')
torch.save([obs, temp], 'data/test_data/nse_data_reg_rbc_test')
