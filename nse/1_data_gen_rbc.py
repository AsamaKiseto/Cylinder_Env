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
N0 = 100
N0 = 1
nx = simulator.params['dimx']
ny = simulator.params['dimy']
dt = simulator.params['dt']
nt = int(4.0 // dt) + 2
nlt = int(2.0 // dt)
nc = int(4.0 // 2.0) + 1
print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}')

temp , velo , p = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny))
ctr = np.linspace(1, 3, N0).reshape(N0, 1).repeat(nc, 1) + (np.random.rand(N0, nc) * 2 - 1) * 0.5
ctr = (np.random.rand(N0, nc) * 2 - 1)
print(ctr.shape, ctr)

for k in range(N0):
    print(f'start # {k}')
    t1 = default_timer()
    simulator.reset(ctr=1.0, const=2.0)

    for init_i in range(int(0.1//dt)):
        simulator.step()
    
    for i in range(nt):
        if i % nlt == 0:
            simulator.set(ctr=ctr[k, i//nlt], const=2.0)
        temp[k, i], velo[k, i], p[k, i], _  = simulator.step()

    t2 = default_timer()
    print(f'# {k} finish | {t2 - t1}')

temp = torch.Tensor(temp).reshape(N0, nt, nx, ny, 1)
velo = torch.Tensor(velo)
p = torch.Tensor(p).reshape(N0, nt, nx, ny, 1)
ctr = torch.Tensor(ctr)

obs = torch.cat((velo, p), dim=-1)

data = [obs, temp, ctr]
# data = [obs, temp]
torch.save(data, 'data/nse_data_reg_rbc_test')


# evaluate
from scripts.draw_utils import *

data_path = 'data/nse_data_reg_rbc_test'
data = LoadDataRBC(data_path)
obs, temp, ctr = data.get_data()
temp = torch.cat((temp, torch.zeros(temp.shape)), dim=-1)
# temp = torch.cat((torch.zeros(temp.shape), temp), dim=-1)
print(obs.shape, temp.shape)

obs_bf = obs[:, :-1]
obs_af = obs[:, 1:]
error = rel_error(obs_bf.reshape(-1, 64, 64, 3), obs_af.reshape(-1, 64, 64, 3))
print(error.mean())

x = np.arange(64) / 64 * 2.0
y = np.arange(64) / 64 * 2.0
x, y = np.meshgrid(x, y)
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)
xy_mesh = [x, y, xl, xh, yl, yh]

animate_field(obs[0, ..., :2], xy_mesh, 'state_6', 'obs', 'rbc')