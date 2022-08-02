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

# env init
env = Cylinder_Rotation_Env(params={'dtr': 0.01, 'T': 1, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 128, 'dimy': 64,
                                    'min_x' : 0,  'max_x' : 2.2, 
                                    'min_y' : 0,  'max_y' : 0.41, 
                                    'r' : 0.05,  'center':(0.2, 0.2),
                                    'min_w': -1, 'max_w': 1,
                                    'min_velocity': -1, 'max_velocity': 1,
                                    'U_max': 1.5, })

if __name__ == '__main__':
    print('start')
    
    # env params
    print(env.params)

    # param setting
    dt = env.params['dtr'] * env.params['T']
    nT = 400
    hf_nT = 200
    nx = env.params['dimx']
    ny = env.params['dimy']
    print(f'dt: {dt} | nt: {nT}')

    # data generate
    Nf = 41
    f1 = np.linspace(-2, 2, Nf)
    f2 = np.linspace(-2, 2, Nf)
    N0 = Nf * Nf
    obs = np.zeros((N0, nT+1, nx, ny, 5), dtype='uint8')
    print(f'state_data_size :{obs.shape}')
    f = np.zeros((N0, nT))
    C_D, C_L, reward = np.zeros((N0, nT)), np.zeros((N0, nT)), np.zeros((N0, nT))
    for k in range(Nf):
        for l in range(Nf):
            print(f'start # {Nf * k + l + 1}')
            start = default_timer()

            obs[Nf * k + l, 0] = env.reset()
        
            for i in range(hf_nT):
                f[Nf * k + l, i] = f1[k]
                obs[Nf * k + l, i+1], reward[Nf * k + l, i], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f1[k])
            
            for i in range(hf_nT, nT):
                f[Nf * k + l, i] = f2[l]
                obs[Nf * k + l, i+1], reward[Nf * k + l, i], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f2[l])
        
            end = default_timer()

            print(f'end # {Nf * k + l + 1} | time: {end-start}')
        # print(f'ang_vel: {ang_v}')
        # print(f'reward :{reward[k]}')

    # np to tensor
    obs_tensor = torch.Tensor(obs)
    reward_tensor = torch.Tensor(reward)
    C_D_tensor = torch.Tensor(C_D)
    C_L_tensor = torch.Tensor(C_L)
    f_tensor = torch.Tensor(f)

    data = [obs_tensor, reward_tensor, C_D_tensor, C_L_tensor, f]

    # save data
    torch.save(data, './data/nse_data_N0_{}_nT_{}'.format(N0, nT))
    