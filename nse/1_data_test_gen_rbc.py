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
N0 = Nf ** 2
nx = simulator.params['dimx']
ny = simulator.params['dimy']
dt = simulator.params['dt']
init_t = 4.0
end_t = 4.0
range_t = 2.0
nlt = int(range_t // dt) + 1
nc = int(init_t // range_t)
nt = int(end_t // dt) + 2
print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}, nlt: {nlt}, nc: {nc}')

temp , velo , p = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny))
ctr = np.linspace(1, 3, N0).reshape(N0, 1).repeat(nc, 1) + (np.random.rand(N0, nc) * 2 - 1) * 2.0
# ctr = np.linspace(0.1, 0.3, N0).reshape(N0, 1).repeat(nc, 1) + (np.random.rand(N0, nc) * 2 - 1) * 0.1
# ctr = 2 * np.random.rand(N0, nc) + 1
ctr1 = np.linspace(1.0, 3.0, Nf).reshape(Nf, 1).repeat(Nf, 1).reshape(N0)
ctr2 = np.linspace(1.0, 3.0, Nf).reshape(1, Nf).repeat(Nf, 0).reshape(N0)
ctr = np.concatenate((ctr1, ctr2), -1)
ctr1 = np.linspace(1.0, 3.0, Nf)
ctr2 = np.linspace(1.0, 3.0, Nf)
ctr1 = np.random.rand(Nf) * 2 + 1
ctr2 = np.random.rand(Nf) * 2 + 1
print(ctr1, ctr2)
# print(ctr.shape, ctr)

for k in range(Nf):
    for l in range(Nf):
        print(f'start # {Nf * k + l + 1}')
        start = default_timer()

        simulator.reset(ctr=1.0, const=0.0)
    
        simulator.set(ctr=ctr1[k], const=0.0)
        for i in range(nlt):
            simulator.step()
        
        simulator.set(ctr=ctr2[l], const=0.0)
        for i in range(nlt):
            simulator.step()
        
        simulator.set(ctr=0.0, const=2.0)
        for i in range(int(nlt)):
            simulator.step()

        for i in range(nt):
            temp[Nf * k + l, i], velo[Nf * k + l, i], p[Nf * k + l, i], _  = simulator.step()
    
        end = default_timer()

        print(f'end # {Nf * k + l + 1} | time: {end-start}')

temp = torch.Tensor(temp).reshape(N0, nt, nx, ny, 1)
velo = torch.Tensor(velo)
p = torch.Tensor(p).reshape(N0, nt, nx, ny, 1)
# ctr = torch.Tensor(ctr).unsqueeze(-1).repeat(1, 1, nlt).reshape(N0, -1)
obs = torch.cat((velo, p), dim=-1)

print(ctr.shape, obs.shape)

# torch.save([obs, temp, ctr], 'data/nse_data_reg_rbc_orig_test')
# torch.save([obs, temp], 'data/nse_data_reg_rbc_test')

# torch.save([obs, temp, ctr], 'data/nse_data_reg_rbc_orig5')
# torch.save([obs, temp], 'data/nse_data_reg_rbc7_1')
torch.save([obs, temp], 'data/test_data/nse_data_reg_rbc7')
