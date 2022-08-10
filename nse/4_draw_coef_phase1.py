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

if __name__ == '__main__':
    # argparser
    args = get_args()
    # path
    data_path = './data/nse_data_N0_256_nT_400'
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
    
    # load logs
    data, _, Cd, Cl, ang_vel  = torch.load(data_path)
    tg = 20     # sample evrey 10 timestamps
    Ng = 1
    data = data[::Ng, ::tg, :, :, 2:]  
    Cd = Cd[::Ng, ::tg]
    Cl = Cl[::Ng, ::tg]
    ang_vel = ang_vel[::Ng, ::tg]

    # data param
    nx = data.shape[2] 
    ny = data.shape[3]
    shape = [nx, ny]
    s = data.shape[2] * data.shape[3]     # ny * nx
    N0 = data.shape[0]                    # num of data sets
    nt = data.shape[1] - 1             # nt
    t = np.arange(nt) * 0.2

    print(ang_vel[k])

    # model
    load_model = FNO_ensemble(model_params, shape, f_channels=f_channels)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False

    data_in = data[k].squeeze()[t_start]
    out_nn = data_in.reshape(nx, ny, 3)
    Cd_nn = torch.zeros(nt)
    Cl_nn = torch.zeros(nt)
    ang_optim = ang_vel[k]

    out_nn = data_in.reshape(1, nx, ny, 3)
    for i in range(t_start, nt):
        ang_nn = ang_optim[i].reshape(1)
        pred, _, _, _ = load_model(out_nn, ang_nn)
        out_nn = pred[:, :, :, :3]
        Cd_nn[i] = torch.mean(pred[:, :, :, -2])
        Cl_nn[i] = torch.mean(pred[:, :, :, -1])

    print(Cd_nn)
    print(Cd[k])

    plt.figure(figsize=(12,10))
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    ax1.set_title('Samples from data', size=15)
    ax1.plot(t[t_start:], Cd[k][t_start:], color='blue')
    ax1.grid(True, lw=0.4, ls="--", c=".50")
    ax1.set_xlim(0, 4)
    # ax1.set_ylim(2.5, 4)
    ax1.set_ylabel(r"$C_d$", fontsize=15)

    ax2.plot(t[t_start:], Cl[k][t_start:], color='blue')
    ax2.grid(True, lw=0.4, ls="--", c=".50")
    ax2.set_ylabel(r"$C_l$", fontsize=15)
    ax2.set_xlabel(r"$t$", fontsize=15)
    ax2.set_xlim(0, 4)
    
    ax1.plot(t[t_start:], Cd_nn[t_start:], color='red')
    ax2.plot(t[t_start:], Cl_nn[t_start:], color='red')

    plt.savefig(f'coef_phase1_#{k}_t_start_{t_start}.jpg')