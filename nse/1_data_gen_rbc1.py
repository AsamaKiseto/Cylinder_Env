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

params = {'dt':  0.05, 'T':  0.01, 'dimx': 64, 'dimy': 64, 'min_x' : 0, 'max_x' : 2.0, 'min_y' : 0.0, 'max_y' : 2.0 ,'Ra':1E6}
simulator = RBC(params)

# N0 = 400 + 1
N0 = 1
nx = simulator.params['dimx']
ny = simulator.params['dimy']
dt = simulator.params['dt']
nt = int(4.0 // dt) + 2
print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}')

temp , velo , p = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny))
# ctr = np.linspace(1, 3, N0)


for k in range(N0):
    print(f'start # {k}')
    t1 = default_timer()
    simulator.reset(ctr=1.0, const=0.0)

    for init_i in range(10 + 4 * k):
        simulator.step()

    simulator.set(ctr=1.0, const=2.0)
    
    for i in range(nt):
        temp[k, i], velo[k, i], p[k, i], _  = simulator.step()

    t2 = default_timer()
    print(f'# {k} finish | {t2 - t1}')

temp = torch.Tensor(temp).reshape(N0, nt, nx, ny, 1)
velo = torch.Tensor(velo)
p = torch.Tensor(p).reshape(N0, nt, nx, ny, 1)
# ctr = torch.Tensor(ctr).reshape(N0, 1).repeat(1, nt)

obs = torch.cat((velo, p), dim=-1)

# data = [obs, temp, ctr]
data = [obs, temp]
torch.save(data, 'data/nse_data_reg_rbc2_test')