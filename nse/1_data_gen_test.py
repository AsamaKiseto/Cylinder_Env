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
    
    parser.add_argument('--f1', default=-3, type=float)
    parser.add_argument('--f2', default=-3, type=float)

    return parser.parse_args(argv)

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
    args = get_args()

    # env params
    print(env.params)

    # param setting
    dt = env.params['dtr'] * env.params['T']
    nT = 600
    hf_nT = int(nT/2)
    nx = env.params['dimx']
    ny = env.params['dimy']
    print(f'dt: {dt} | nt: {nT}')

    # data generate
    t = np.arange(nT)
    f = 0.1 * (np.sin(1.5*t) + 0.15) * (0.1 * np.sin(0.5*t) + 1)
    print(f'f: {f}')
    N0 = 1
    obs = np.zeros((nT+1, nx, ny, 5))
    print(f'state_data_size :{obs.shape}')
    C_D, C_L, reward = np.zeros((nT)), np.zeros((nT)), np.zeros((nT))

    print('start')
    start = default_timer()

    obs[0] = env.reset()

    for i in range(nT):
        obs[i+1], reward[i], C_D[i], C_L[i] = env.step(f[i])
        if((i+1) % 20 == 0):
            print(f'# {i+1}')
    

    end = default_timer()

    print(f'end | time: {end-start}')

    # np to tensor
    obs_tensor = torch.Tensor(obs)
    reward_tensor = torch.Tensor(reward)
    C_D_tensor = torch.Tensor(C_D)
    C_L_tensor = torch.Tensor(C_L)
    f_tensor = torch.Tensor(f)

    data = [obs_tensor, reward_tensor, C_D_tensor, C_L_tensor, f_tensor]

    # save data
    torch.save(data, './data/nse_data_test'.format(f[0]))
    