import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 
from timeit import default_timer

import argparse

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    parser.add_argument('-dn', '--data_name', default="test", type=str)
    parser.add_argument('-Tr', '--Tr', default=4, type=float)
    parser.add_argument('-fr', '--f_range', default=1, type=float)
    parser.add_argument('-fb', '--f_base', default=0, type=float)
    parser.add_argument('-dt', '--dt', default=0.01, type=float)
    parser.add_argument('-Nf', '--Nf', default=8, type=int)

    return parser.parse_args(argv)

print('start')
args = get_args()

# env init
env = Cylinder_Rotation_Env(params={'dt': args.dt, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 256, 'dimy': 64,
                                    'min_x' : 0,  'max_x' : 2.2, 
                                    'min_y' : 0,  'max_y' : 0.41, 
                                    'r' : 0.05,  'center':(0.2, 0.2),
                                    'min_w': -1, 'max_w': 1,
                                    'min_velocity': -1, 'max_velocity': 1,
                                    'U_max': 1.5, })

# env params
print(env.params)

# param setting
dt = env.params['dt']
Tr = args.Tr
nT = int (Tr / dt)
hf_nT = int(nT / 2)
nx = env.params['dimx']
ny = env.params['dimy']
print(f'dt: {dt} | nt: {nT}')

# data generate
Nf = args.Nf + 1
fr = args.f_range
fb = args.f_base
f1 = np.linspace(-fr + fb, fr + fb, Nf)
f2 = np.linspace(-fr + fb, fr + fb, Nf)
print(f'f1: {f1}')
print(f'f2: {f2}')
N0 = Nf * Nf
obs = np.zeros((N0, nT+1, nx, ny, 5))
print(f'state_data_size :{obs.shape}')
f = np.zeros((N0, nT))
C_D, C_L = np.zeros((N0, nT)), np.zeros((N0, nT))

# env init step
start = default_timer()
nT_init = int(4 / dt)
for i in range(nT_init):
    env.step(0.00)
end = default_timer()
print(f'init complete: {end - start}')

env.set_init()
# obs_temp = env.reset(mode='vertex')
# obs_temp = env.reset(mode='grid')
# mesh_num = obs_temp.shape[0]
# obs_v = np.zeros((N0, nT+1, mesh_num, 5))

for k in range(Nf):
    for l in range(Nf):
        print(f'start # {Nf * k + l + 1}')
        start = default_timer()

        # obs_v[Nf * k + l, 0] = env.reset(mode='vertex')
        obs[Nf * k + l, 0] = env.reset(mode='grid')
    
        for i in range(hf_nT):
            f[Nf * k + l, i] = f1[k]
            obs[Nf * k + l, i+1], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f[Nf * k + l, i])
            # obs_v[Nf * k + l, i+1], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f1[k], mode='vertex')
        
        for i in range(hf_nT, nT):
            f[Nf * k + l, i] = f2[l]
            obs[Nf * k + l, i+1], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f[Nf * k + l, i])
            # obs_v[Nf * k + l, i+1], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f2[l], mode='vertex')
    
        end = default_timer()

        print(f'end # {Nf * k + l + 1} | time: {end-start}')
    # print(f'ang_vel: {ang_v}')
    # print(f'reward :{reward[k]}')

# np to tensor
obs_tensor = torch.Tensor(obs)
# obs_v_tensor = torch.Tensor(obs_v)
C_D_tensor = torch.Tensor(C_D)
C_L_tensor = torch.Tensor(C_L)
ctr_tensor = torch.Tensor(f)

data = [obs_tensor, C_D_tensor, C_L_tensor, ctr_tensor]
# data_v = [obs_v_tensor, C_D_tensor, C_L_tensor, f_tensor]

# save data
# torch.save(data, './data/nse_data_N0_{}_nT_{}_f1_{}_f2_{}'.format(N0, nT, args.f1, args.f2))
# torch.save(data_v, './data/nse_data_irr')
torch.save(data, f'./data/nse_data_reg_dt_{dt}_fb_{args.f_base}_fr_{args.f_range}')
# torch.save(data, f'./data/test_data/nse_data_reg_scale_{args.scale}_{args.data_name}')
# torch.save(data, './data/nse_data_test1')