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

params = {'dt':  0.01, 'T':  0.01, 'dimx': 64, 'dimy': 32, 'min_x' : 0, 'max_x' : 2.0, 'min_y' : 0.0, 'max_y' : 1.0 ,'Ra':1E2}
simulator = RBC(params)

Nf = 10
# Nf = 1
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
ctr1 = np.random.rand(Nf) * 4 + 1
ctr2 = np.random.rand(Nf) * 4 + 1
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

# torch.save([obs, temp, ctr], 'data/nse_data_reg_rbc_orig5')
torch.save([obs, temp, ctr], 'data/test_data/nse_data_reg_rbc7')
# torch.save([obs, temp], 'data/test_data/nse_data_reg_rbc7')

# evaluate
from scripts.draw_utils import *

data_path = 'data/test_data/nse_data_reg_rbc7'
data = LoadDataRBC(data_path)
obs, temp, ctr = data.get_data()
temp = torch.cat((temp, torch.zeros(temp.shape)), dim=-1)
# temp = torch.cat((torch.zeros(temp.shape), temp), dim=-1)
print(obs.shape, temp.shape)

obs_bf = obs[:, :-1]
obs_af = obs[:, 1:]
error = rel_error(obs_bf.reshape(-1, 64, 32, 3), obs_af.reshape(-1, 64, 32, 3))
print(error)
print(error.mean())

# x = np.arange(64) / 64 * 2.0
# y = np.arange(32) / 32 * 1.0
# x, y = np.meshgrid(x, y)
# xl, xh  = np.min(x), np.max(x)
# yl, yh = np.min(y), np.max(y)
# xy_mesh = [x, y, xl, xh, yl, yh]

# animate_field(obs[0, ..., :2], xy_mesh, 'state_5', 'obs', 'rbc')

# # rerun
# from scripts.draw_utils import *
# data_path = 'data/nse_data_reg_rbc_orig_test'
# data = LoadDataRBC(data_path)
# obs, temp, ctr = data.get_data()
# ctr = ctr[:, ::nlt]
# ctr = ctr[0].unsqueeze(0)
# temp , velo , p = np.zeros((N0, nt, nx, ny)), np.zeros((N0, nt, nx, ny, 2)), np.zeros((N0, nt, nx, ny))

# for k in range(N0):
#     print(f'start # {k}')
#     t1 = default_timer()
#     simulator.reset(ctr=1.0, const=0.0)

#     for i in range(init_nt):
#         if i % nlt == 0 and i != init_nt-1:
#             simulator.set(ctr=ctr[k, i//nlt], const=0.0)
#         simulator.step()
    
#     simulator.set(ctr=0.1, const=0.0)
    
#     for i in range(nt):
#         temp[k, i], velo[k, i], p[k, i], _  = simulator.step()

#     t2 = default_timer()
#     print(f'# {k} finish | {t2 - t1}')

# temp = torch.Tensor(temp).reshape(N0, nt, nx, ny, 1)
# velo = torch.Tensor(velo)
# p = torch.Tensor(p).reshape(N0, nt, nx, ny, 1)
# ctr = torch.Tensor(ctr).unsqueeze(-1).repeat(1, 1, nlt).reshape(N0, -1)
# obs = torch.cat((velo, p), dim=-1)

# print(ctr.shape, obs.shape)

# torch.save([obs, temp], 'data/nse_data_reg_rbc_test1')

# # evaluate
# from scripts.draw_utils import *
# data_path = 'data/nse_data_reg_rbc_test1'
# data = LoadDataRBC1(data_path)
# obs, temp, ctr = data.get_data()
# temp = torch.cat((temp, torch.zeros(temp.shape)), dim=-1)
# # temp = torch.cat((torch.zeros(temp.shape), temp), dim=-1)
# print(obs.shape, temp.shape)

# obs_bf = obs[:, :-1]
# obs_af = obs[:, 1:]
# error = rel_error(obs_bf.reshape(-1, 64, 32, 3), obs_af.reshape(-1, 64, 32, 3))
# print(error.mean())

# x = np.arange(64) / 64 * 2.0
# y = np.arange(32) / 32 * 1.0
# x, y = np.meshgrid(x, y)
# xl, xh  = np.min(x), np.max(x)
# yl, yh = np.min(y), np.max(y)
# xy_mesh = [x, y, xl, xh, yl, yh]

# animate_field(obs[0, ..., :2], xy_mesh, 'state_2_1', 'obs', 'rbc')