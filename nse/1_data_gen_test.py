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
    parser.add_argument('-fb', '--f_base', default=-1, type=float)
    parser.add_argument('-s', '--scale', default=0, type=float)
    parser.add_argument('-N0', '--N0', default=10, type=int)

    return parser.parse_args(argv)

# env init
env = Cylinder_Rotation_Env(params={'dt': 0.01, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 256, 'dimy': 64,
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
    dt = env.params['dt']
    nT = 400
    hf_nT = int(nT/2)
    nx = env.params['dimx']
    ny = env.params['dimy']
    print(f'dt: {dt} | nt: {nT}')

    # data generate
    N0 = args.N0
    tg = 5
    nt = int(nT / tg)

    obs = np.zeros((N0, nT+1, nx, ny, 5))
    print(f'state_data_size :{obs.shape}')
    ctr = np.zeros((N0, nT))
    C_D, C_L = np.zeros((N0, nT)), np.zeros((N0, nT))

    # env init step
    start = default_timer()
    nT_init = 10
    for i in range(nT_init):
        env.step(0.00)
    end = default_timer()
    print(f'init complete: {end - start}')

    env.set_init()
    # obs_temp = env.reset(mode='vertex')
    # obs_temp = env.reset(mode='grid')
    # mesh_num = obs_temp.shape[0]
    # obs_v = np.zeros((N0, nT+1, mesh_num, 5))

    for k in range(N0):
        ctr_temp = args.scale * (np.random.rand(nt) - 0.5) + args.f_base
        ctr_temp = ctr_temp.reshape(nt, 1).repeat(tg, 1).reshape(-1)
        ctr[k] = ctr_temp

        print(f'start # {k}')
        start = default_timer()

        # obs_v[Nf * k + l, 0] = env.reset(mode='vertex')
        obs[k, 0] = env.reset(mode='grid')
    
        for i in range(nT):
            obs[k, i+1], C_D[k, i], C_L[k, i] = env.step(ctr_temp[i])
            # obs_v[Nf * k + l, i+1], C_D[Nf * k + l, i], C_L[Nf * k + l, i] = env.step(f1[k], mode='vertex')
    
        end = default_timer()

        print(f'end # {k} | time: {end-start}')
        # print(f'ang_vel: {ang_v}')
        # print(f'reward :{reward[k]}')

    # np to tensor
    obs_tensor = torch.Tensor(obs)
    # obs_v_tensor = torch.Tensor(obs_v)
    C_D_tensor = torch.Tensor(C_D)
    C_L_tensor = torch.Tensor(C_L)
    ctr_tensor = torch.Tensor(ctr)

    data = [obs_tensor, C_D_tensor, C_L_tensor, ctr_tensor]
    # data_v = [obs_v_tensor, C_D_tensor, C_L_tensor, f_tensor]

    # save data
    # torch.save(data, './data/nse_data_N0_{}_nT_{}_f1_{}_f2_{}'.format(N0, nT, args.f1, args.f2))
    # torch.save(data_v, './data/nse_data_irr')
    # torch.save(data, './data/nse_data_reg_extra')
    torch.save(data, f'./data/test_data/nse_data_reg_scale_{args.scale}_{args.data_name}')
    # torch.save(data, './data/nse_data_test1')