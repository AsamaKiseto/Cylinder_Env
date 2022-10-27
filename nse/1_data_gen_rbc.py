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

N0 = 10
nx = simulator.params['dimx']
ny = simulator.params['dimy']
dt = simulator.params['dt']
init_nt = int(10 // dt)
nt = int(10 // dt)
print(f'N0: {N0}, nt: {nt}, nx: {nx}, ny: {ny}')

temp , velo , p , a1 , b1  = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt)), np.zeros((N0, nt))

for k in range(N0):
    print(f'start # {k}')
    t1 = default_timer()
    simulator.reset()

    for i in range(nt):
        temp[k, i], velo[k, i], _, p[k, i], a1[k, i], b1[k, i] = simulator.step()

    t2 = default_timer()
    print(f'# {k} finish | {t2 - t1}')

temp = torch.Tensor(temp)
velo = torch.Tensor(velo)
p = torch.Tensor(p)
a1 = torch.Tensor(a1)
b1 = torch.Tensor(b1)

obs = torch.cat((velo, p), dim=-1)

data = [temp , velo , p , a1 , b1]
torch.save(data, 'data/rbc_data_reg')