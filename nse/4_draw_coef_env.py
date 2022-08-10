import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 
from timeit import default_timer

from models import *
from utils import *

# plot colors
from matplotlib import colors

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    
    parser.add_argument('--operator_path', default='phase1_logs_ex12', type=str, help='path of operator weight')
    parser.add_argument('--t_start', default=1, type=int, help='data number')
    parser.add_argument('--k', default=0, type=int)

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

print(env.params)


if __name__ == '__main__':
    # argparser
    args = get_args()

    # path
    operator_path = 'logs/' + args.operator_path

    # mosel params setting
    state_dict, logs = torch.load(operator_path)
    params_args = logs['args']
    L = params_args.L
    modes = params_args.modes
    width = params_args.width
    model_params = dict()
    model_params['modes'] = modes
    model_params['width'] = width
    model_params['L'] = L
    f_channels = params_args.f_channels

    t_start = args.t_start
    k = args.k  # k th traj
    
    # data param
    nx, ny = 128, 64
    shape = [nx, ny]
    nt = 20
    tg = 20
    nT = nt * tg

    # f = np.random.rand(nt) * 4 - 2
    f = np.arange(nt) / nt * 4 - 2
    f_nn = torch.Tensor(f)
    print(f)

    obs_nn = torch.zeros(nt, nx, ny, 3)
    Cd_nn = torch.zeros(nt)
    Cl_nn = torch.zeros(nt)

    obs = np.zeros((nT+1, nx, ny, 5))
    Cd = np.zeros(nT)
    Cl = np.zeros(nT)

    t_nn = (np.arange(nt) + 1) * 0.2
    t = (np.arange(nt * tg)) * 0.01 
    
    # model
    load_model = FNO_ensemble(model_params, shape, f_channels=f_channels)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False

    obs[0] = env.reset()
    for i in range(t_start):
        for j in range(tg):
            obs[i*tg + j + 1], _, Cd[i*tg + j], Cl[i*tg + j] = env.step(f[i])

    out_nn = torch.Tensor(obs[tg * t_start, :, :, 2:]).reshape(1, nx, ny, 3)
    for i in range(t_start, nt):
        print(f'start #{i} f: {f[i]}')
        for j in range(tg):
            obs[i*tg + j + 1], _, Cd[i*tg + j], Cl[i*tg + j] = env.step(f[i])
        pred, _, _, _ = load_model(out_nn, f_nn[i].reshape(1))
        out_nn = pred[:, :, :, :3]
        Cd_nn[i] = torch.mean(pred[:, :, :, -2])
        Cl_nn[i] = torch.mean(pred[:, :, :, -1])
    
    torch.save([obs, Cd, Cl, obs_nn, Cd_nn, Cl_nn], 'logs/phase1_env_logs')

    plt.figure(figsize=(12,10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax1.set_title('Samples from data', size=15)
    ax1.plot(t, Cd, color='yellow')
    ax1.grid(True, lw=0.4, ls="--", c=".50")
    ax1.set_xlim(0, 4)
    # ax1.set_ylim(2.5, 4)
    ax1.set_ylabel(r"$C_d$", fontsize=15)

    ax2.plot(t, Cl, color='yellow')
    ax2.grid(True, lw=0.4, ls="--", c=".50")
    ax2.set_ylabel(r"$C_l$", fontsize=15)
    ax2.set_xlabel(r"$t$", fontsize=15)
    ax2.set_xlim(0, 4)
    
    ax1.plot(t_nn[t_start:], Cd_nn[t_start:], color='red')
    ax2.plot(t_nn[t_start:], Cl_nn[t_start:], color='red')

    ax1.plot(t_nn, Cd[(tg-1)::tg], color='blue')
    ax2.plot(t_nn, Cl[(tg-1)::tg], color='blue')

    plt.savefig(f'coef_phase1.jpg')