import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 
from timeit import default_timer

from scripts.models import *
from scripts.utils import *

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
    
    parser.add_argument('-op', '--operator_path', default='ex3', type=str, help='path of operator weight')
    parser.add_argument('-s', '--scale', default=1, type=float, help='random scale')
    parser.add_argument('--t_start', default=5, type=int, help='data number')

    return parser.parse_args(argv)

# env init
env = Cylinder_Rotation_Env(params={'dtr': 0.01, 'T': 1, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 256, 'dimy': 64,
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
    operator_path = 'logs/phase1_' + args.operator_path + '_grid_pi'
    modify = True
    logs_path = 'logs/phase1_env_logs_' + args.operator_path

    # logs
    if not os.path.isfile(logs_path):
        logs=dict()

        logs['scale']=[]
        logs['t_start']=[]

        logs['obs']=[]
        logs['Cd']=[]
        logs['Cl']=[]

        logs['obs_nn']=[]
        logs['Cd_nn']=[]
        logs['Cl_nn']=[]
        logs['Lpde_nn']=[]
        
        logs['Cd_var']=[]
        logs['Cl_var']=[]
        logs['obs_var']=[]
    else:
        logs = torch.load(logs_path)

    # mosel params setting
    state_dict, logs = torch.load(operator_path)
    params_args = logs['args']
    L = params_args.L
    modes = params_args.modes
    width = params_args.width
    tg = params_args.tg
    f_channels = params_args.f_channels
    model_params = dict()
    model_params['modes'] = modes
    model_params['width'] = width
    model_params['L'] = L
    model_params['f_channels'] = f_channels

    Cd_mean, Cd_var = logs['data_norm']['Cd']
    Cl_mean, Cl_var = logs['data_norm']['Cl']
    ctr_mean, ctr_var = logs['data_norm']['ctr']

    t_start = args.t_start
    scale = args.scale
    logs['t_start'].append(args.t_start)
    logs['scale'].append(args.scale)
    
    # data param
    nx, ny = env.params['dimx'], env.params['dimy']
    shape = [nx, ny]
    model_params['shape'] = shape
    dt = env.params['dtr'] * env.params['T']

    nT = 400
    nt = nT // tg

    t_nn = (np.arange(nt)) * 0.01 * tg
    t = (np.arange(nt * tg) + 1) * 0.01 

    f = scale * (np.random.rand(nt) - 0.5)
    # f = np.ones(nt) * (-3)
    f_nn = torch.Tensor(f)
    print(f)

    obs_nn = torch.zeros(nt, nx, ny, 3)
    Cd_nn = torch.zeros(nt)
    Cl_nn = torch.zeros(nt)
    Lpde_nn = torch.zeros(nt)

    obs = np.zeros((nT+1, nx, ny, 5))
    Cd = np.zeros(nT)
    Cl = np.zeros(nT)

    # model
    load_model = FNO_ensemble(model_params)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False

    obs[0] = env.reset()
    for i in range(t_start):
        print(f'# {i+1} f: {f[i]}')
        for j in range(tg):
            obs[i*tg + j + 1], Cd[i*tg + j], Cl[i*tg + j] = env.step(f[i])
        obs_nn[i] = torch.Tensor(obs[i*tg + tg, ..., 2:])

    out_nn = torch.Tensor(obs[tg * t_start, :, :, 2:]).reshape(1, nx, ny, 3)
    for i in range(t_start, nt):
        print(f'# {i+1} f: {f[i]}')
        for j in range(tg):
            obs[i*tg + j + 1], Cd[i*tg + j], Cl[i*tg + j] = env.step(f[i])
        pred, _, _, _ = load_model(out_nn, f_nn[i].reshape(1), modify)
        bf_mod = load_model.state_mo(out_nn, modify)
        bf = bf_mod + out_nn
        out_nn = pred[:, :, :, :3]
        out_mod = load_model.state_mo(out_nn, modify)
        af = out_mod + out_nn
        Lpde_nn[i] = Lpde(af, bf, dt*tg)
        # out_nn = out_nn + out_mod
        obs_nn[i] = out_nn
        Cd_nn[i] = torch.mean(pred[:, :, :, -2])
        Cl_nn[i] = torch.mean(pred[:, :, :, -1])
        print(Cd_nn[i], Cl_nn[i], Lpde_nn[i])

    Cd_nn = Cd_nn * Cd_var + Cd_mean
    Cl_nn = Cl_nn * Cl_var + Cl_mean

    Cd_nn, Cl_nn, obs_nn = Cd_nn.detach().numpy(), Cl_nn.detach().numpy(), obs_nn.detach().numpy()
    Cd_sps, Cl_sps, obs_sps = Cd[tg-1::tg], Cl[tg-1::tg], obs[::tg][1:][...,2:]

    Cd_nn[:t_start] = Cd_sps[:t_start]
    Cl_nn[:t_start] = Cl_sps[:t_start]

    print(Cd_sps.shape, obs_sps.shape)
    print(Cd_nn.shape, obs_nn.shape)

    Cd_var = Cd_sps - Cd_nn
    Cl_var = Cl_sps - Cl_nn
    obs_var = obs_sps - obs_nn
    Cd_var = np.mean((Cd_var.reshape(nt, -1)**2), 1)
    Cl_var = np.mean((Cl_var.reshape(nt, -1)**2), 1)
    obs_var = np.mean((obs_var.reshape(nt, -1)**2), 1)
    
    logs['obs'].append(obs)
    logs['Cd'].append(Cd)
    logs['Cl'].append(Cl)
    logs['obs_nn'].append(obs_nn)
    logs['Cd_nn'].append(Cd_nn)
    logs['Cl_nn'].append(Cl_nn)
    logs['Lpde_nn'].append(Lpde_nn)

    logs['Cd_var'].append(Cd_var)
    logs['Cl_var'].append(Cl_var)
    logs['obs_var'].append(obs_var)

    # torch.save([obs, Cd, Cl, obs_nn, Cd_nn, Cl_nn, Lpde_nn], 'logs/phase1_env_logs_scale_{}'.format(scale))

    # dt = 0.01
    # tg = 5
    # t_start = 10
    # nT = 400
    # nt = nT // tg
    # dt = 0.01

    # t_nn = (np.arange(nt)) * 0.01 * tg
    # t = (np.arange(nt * tg) + 1) * 0.01 

    # obs, Cd, Cl, obs_nn, Cd_nn, Cl_nn, Lpde_nn = torch.load( 'logs/phase1_env_logs_scale_{}'.format(scale))
    plt.figure(figsize=(15,12))
    ax1 = plt.subplot2grid((4, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((4, 2), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((4, 2), (2, 0), colspan=2)
    ax4 = plt.subplot2grid((4, 2), (3, 0), colspan=2)

    ax1.grid(True, lw=0.4, ls="--", c=".50")
    ax1.set_xlim(0, nt * tg * dt)
    ax2.grid(True, lw=0.4, ls="--", c=".50")
    ax2.set_xlim(0, nt * tg * dt)
    ax3.grid(True, lw=0.4, ls="--", c=".50")
    ax3.set_xlim(0, nt * tg * dt)
    ax4.grid(True, lw=0.4, ls="--", c=".50")
    ax4.set_xlim(0, nt * tg * dt)

    ax1.set_title(r"$error/loss in different scales$", fontsize=15)
    ax1.set_ylabel(r"$C_d$", fontsize=15)
    ax2.set_ylabel(r"$C_l$", fontsize=15)
    ax3.set_ylabel(r"$state$", fontsize=15)
    ax4.set_ylabel(r"$L_{pde}$", fontsize=15)
    ax4.set_xlabel(r"$t$", fontsize=15)
    
    ax1.plot(t, Cd, color='blue')
    ax2.plot(t, Cl, color='blue')

    ax1.plot(t_nn[t_start:], Cd_nn[t_start:], color='red')
    ax2.plot(t_nn[t_start:], Cl_nn[t_start:], color='red')

    ax1.plot(t_nn, Cd_sps, color='yellow')
    ax2.plot(t_nn, Cl_sps, color='yellow')

    ax3.plot(t_nn, obs_var)
    ax4.plot(t_nn, Lpde_nn)
    
    ax1.set_ylim(2.5, 3.5)
    ax2.set_ylim(-1.5, 1.5)
    ax3.set_ylim(0, 1)
    ax4.set_ylim(0, 1)

    plt.savefig(f'logs/coef_phase1_{args.operator_path}_scale_{scale}.jpg')