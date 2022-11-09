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

simulator = RBC()

N0 = 100 + 1
nx = simulator.params['dimx']
ny = simulator.params['dimy']
dt = simulator.params['dt']
nt = int(1 // dt) + 2
print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}')

N0 = 3
temp , velo , p = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny))
ctr = np.linspace(-1, 1, N0)
# ctr = np.zeros(N0)
ctr = np.array([0, 1, 2])

for k in range(N0):
    print(f'start # {k}')
    t1 = default_timer()
    simulator.reset(ctr[k], const=1.0)

    for init_i in range(2):
        simulator.step()

    for i in range(nt):
        temp[k, i], velo[k, i], p[k, i], _  = simulator.step()

    t2 = default_timer()
    print(f'# {k} finish | {t2 - t1}')

temp = torch.Tensor(temp).reshape(N0, nt, nx, ny, 1)
velo = torch.Tensor(velo)
p = torch.Tensor(p).reshape(N0, nt, nx, ny, 1)
ctr = torch.Tensor(ctr).reshape(N0, 1).repeat(1, nt)

obs = torch.cat((velo, p), dim=-1)

data = [obs, temp , ctr]
torch.save(data, 'data/nse_data_reg_rbc_test')