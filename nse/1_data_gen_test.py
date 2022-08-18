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
    nT_init = 300
    nT = nT_init + 100
    nx = env.params['dimx']
    ny = env.params['dimy']
    print(f'dt: {dt} | nt: {nT}')

    # data generate
    N0 = 10
    t = np.arange(nT)
    f = np.linspace(-5, 5, N0)
    print(f'f: {f}')
    
    obs = np.zeros((N0, nT+1, nx, ny, 5))
    print(f'state_data_size :{obs.shape}')
    Cd, Cl = np.zeros((N0, nT)), np.zeros((N0, nT))

    print('start')
    obs_init = np.zeros((nT_init+1, nx, ny, 5))
    Cd_init, Cl_init = np.zeros(nT_init), np.zeros(nT_init)
    obs_init[0] = env.reset()

    start = default_timer()
    for i in range(nT_init):
        obs_init[i+1], Cd_init[i], Cl_init[i] = env.step(0.00)
    end = default_timer()
    print(f'init complete: {end - start}')

    env.set_init()
    
    for k in range(N0):
        print(f'# {k}')
        start = default_timer()
        
        obs[k, :nT_init+1], Cd[k, :nT_init], Cl[k, :nT_init] = obs_init, Cd_init, Cl_init
        env.reset()

        for i in range(nT_init, nT):
            obs[k, i+1], Cd[k, i], Cl[k, i] = env.step(f[k])

        end = default_timer()
        print(f'end | time: {end - start}')

    # np to tensor
    obs_tensor = torch.Tensor(obs)
    Cd_tensor = torch.Tensor(Cd)
    Cl_tensor = torch.Tensor(Cl)
    f_tensor = torch.Tensor(f)

    data = [obs_tensor, Cd_tensor, Cl_tensor, f_tensor]

    # save data
    torch.save(data, 'data/nse_data_test')
    