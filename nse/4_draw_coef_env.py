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
from scripts.draw_utils import *

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    
    parser.add_argument('-op', '--operator', default='ex0', type=str, help='path of operator weight')
    parser.add_argument('-nt', '--nt', default=2, type=int, help='nums of timestamps')
    parser.add_argument('-tg', '--tg', default=5, type=int, help='gap of timestamps')
    parser.add_argument('-s', '--scale', default=0, type=float, help='random scale')
    parser.add_argument('-ts', '--t_start', default=0, type=int, help='data number')

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


class load_model():
    def __init__(self, operator_path, shape):
        # mosel params setting
        print(operator_path)
        state_dict, logs_model = torch.load(operator_path)
        self.modify = logs_model['modify']
        # self.modify = True
        self.data_norm = logs_model['data_norm']
        params_args = logs_model['args']
        L = params_args.L
        modes = params_args.modes
        width = params_args.width
        self.tg = params_args.tg
        self.dt = self.tg * 0.01
        f_channels = params_args.f_channels

        model_params = dict()
        model_params['modes'] = modes
        model_params['width'] = width
        model_params['L'] = L
        model_params['f_channels'] = f_channels
        model_params['shape'] = shape

        self.model = FNO_ensemble(model_params)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.logs = dict()
        self.logs['obs_nn']=[]
        self.logs['Cd_nn']=[]
        self.logs['Cl_nn']=[]
        self.logs['Lpde_nn']=[]
    
    def step(self, f_nn):
        pred, _, _, _, mod = self.model(self.in_nn, f_nn.reshape(1), self.modify)
        bf = self.in_nn
        out_nn = pred[:, :, :, :3]
        af = out_nn
        self.in_nn = out_nn
        Lpde_nn = ((Lpde(af, bf, self.dt) + mod) ** 2).mean()
        Cd_nn = torch.mean(pred[:, :, :, -2])
        Cl_nn = torch.mean(pred[:, :, :, -1])

        Cd_mean, Cd_var = self.data_norm['Cd']
        Cl_mean, Cl_var = self.data_norm['Cl']
        Cd_nn = Cd_nn * Cd_var + Cd_mean
        Cl_nn = Cl_nn * Cl_var + Cl_mean

        self.logs['Cd_nn'].append(Cd_nn.detach().numpy())
        self.logs['Cl_nn'].append(Cl_nn.detach().numpy())
        self.logs['obs_nn'].append(out_nn.squeeze().detach().numpy())
        self.logs['Lpde_nn'].append(Lpde_nn.detach().numpy())
    
    def set_init(self, state_nn):
        self.in_nn = state_nn

    def combine_data(self, data_ts):
        obs_ts, Cd_ts, Cl_ts = data_ts
        Lpde_ts = np.zeros(t_start)
        self.obs_nn = np.concatenate((obs_ts, np.asarray(self.logs['obs_nn'])), 0)
        self.Cd_nn = np.concatenate((Cd_ts, np.asarray(self.logs['Cd_nn'])), 0)
        self.Cl_nn = np.concatenate((Cl_ts, np.asarray(self.logs['Cl_nn'])), 0)
        self.Lpde_nn = np.concatenate((Lpde_ts, np.asarray(self.logs['Lpde_nn'])), 0)

    def compute_error(self, data_sps):
        obs_sps, Cd_sps, Cl_sps = data_sps
        Cd_var = Cd_sps - self.Cd_nn
        Cl_var = Cl_sps - self.Cl_nn
        obs_var = obs_sps - self.obs_nn
        self.Cd_var = np.mean((Cd_var.reshape(nt, -1)**2), 1)
        self.Cl_var = np.mean((Cl_var.reshape(nt, -1)**2), 1)
        self.obs_var = np.mean((obs_var.reshape(nt, -1)**2), 1)

    def plot(self, ax, t_nn, label=None):
        ax[0].plot(t_nn, self.Cd_nn, label=f'{label}')
        ax[1].plot(t_nn, self.Cl_nn, label=f'{label}')
        ax[2].plot(t_nn, self.obs_var, label=f'{label}')
        ax[3].plot(t_nn, self.Lpde_nn, label=f'{label}')
        for i in range(4):
            ax[i].legend()
    
    def save_logs(self, logs):
        logs['obs_nn'].append(self.obs_nn)
        logs['Cd_nn'].append(self.Cd_nn)
        logs['Cl_nn'].append(self.Cl_nn)
        logs['Lpde_nn'].append(self.Lpde_nn)
        logs['Cd_var'].append(self.Cd_var)
        logs['Cl_var'].append(self.Cl_var)
        logs['obs_var'].append(self.obs_var)

if __name__ == '__main__':
    # argparser
    args = get_args()

    # path
    logs_path = 'logs/phase1_env_logs'

    ex_nums = ['ex0', 'ex7', 'ex7_nomod']
    # ex_nums = ['ex0', 'ex7']
    # ex_nums = ['ex0_big', 'ex3_big', 'ex3_big_nomod']
    label = ['without_pde_loss', 'with_modify', 'without_modify']
    # label = ['without_pde_loss', 'with_modify']
    n_model = len(ex_nums)

    # logs
    if not os.path.isfile(logs_path):
        logs=dict()

        logs['ex_nums'] = ex_nums
        logs['label'] = label
        logs['scale']=[]
        logs['t_start']=[]
        logs['obs']=[]
        logs['Cd']=[]
        logs['Cl']=[]

        for i in range(n_model):
            logs[ex_nums[i]] = dict()
            logs[ex_nums[i]]['obs_nn']=[]
            logs[ex_nums[i]]['Cd_nn']=[]
            logs[ex_nums[i]]['Cl_nn']=[]
            logs[ex_nums[i]]['Lpde_nn']=[]
            logs[ex_nums[i]]['Cd_var']=[]
            logs[ex_nums[i]]['Cl_var']=[]
            logs[ex_nums[i]]['obs_var']=[]

    else:
        logs = torch.load(logs_path)

    # data param
    nx, ny = env.params['dimx'], env.params['dimy']
    shape = [nx, ny]
    dt = env.params['dtr'] * env.params['T']

    t_start = args.t_start
    scale = args.scale
    logs['t_start'].append(t_start)
    logs['scale'].append(scale)
    
    # data 
    tg = args.tg
    nt = args.nt
    nT = nt * tg
    t_nn = (np.arange(nt) + 1) * 0.01 * tg
    t = (np.arange(nt * tg) + 1) * 0.01 
    f = scale * (np.random.rand(nt) - 0.5)
    # f = np.ones(nt)
    f_nn = torch.Tensor(f)
    print(f)

    obs_ts = np.zeros((t_start, nx, ny, 3))
    Cd_ts = np.zeros(t_start)
    Cl_ts = np.zeros(t_start)

    obs = np.zeros((nT+1, nx, ny, 5))
    Cd = np.zeros(nT)
    Cl = np.zeros(nT)

    # env init step
    start = default_timer()
    nT_init = 10
    for i in range(nT_init):
        env.step(0.00)
    end = default_timer()
    print(f'init complete: {end - start}')
    env.set_init()
    obs[0] = env.reset()
    # env step
    for i in range(t_start):
        print(f'# {i+1} f: {f[i]}')
        for j in range(tg):
            obs[i*tg + j + 1], Cd[i*tg + j], Cl[i*tg + j] = env.step(f[i])
    in_nn = torch.Tensor(obs[tg * t_start, :, :, 2:]).reshape(1, nx, ny, 3)
    for i in range(t_start, nt):
        print(f'# {i+1} f: {f[i]}')
        for j in range(tg):
            obs[i*tg + j + 1], Cd[i*tg + j], Cl[i*tg + j] = env.step(f[i])
    
    obs_sps, Cd_sps, Cl_sps = obs[::tg][1:][...,2:], Cd[tg-1::tg], Cl[tg-1::tg]
    obs_ts, Cd_ts, Cl_ts = obs_sps[:t_start], Cd_sps[:t_start], Cl_sps[:t_start]
    data_ts = [obs_ts, Cd_ts, Cl_ts]
    data_sps = [obs_sps, Cd_sps, Cl_sps]

    logs['obs'].append(obs)
    logs['Cd'].append(Cd)
    logs['Cl'].append(Cl)

    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_xlim(0, nt * tg * dt)
        
    ax[0].set_title(r"$error/loss in different scales$", fontsize=15)
    ax[0].set_ylabel(r"$C_d$", fontsize=15)
    ax[1].set_ylabel(r"$C_l$", fontsize=15)
    ax[2].set_ylabel(r"$state$", fontsize=15)
    ax[3].set_ylabel(r"$L_{pde}$", fontsize=15)
    ax[3].set_xlabel(r"$t$", fontsize=15)
    
    ax[0].plot(t, Cd, color='blue')
    ax[1].plot(t, Cl, color='blue')

    ax[0].plot(t_nn, Cd_sps, color='yellow')
    ax[1].plot(t_nn, Cl_sps, color='yellow')

    # mosel setting
    for i in range(n_model):
        operator_path = 'logs/phase1_' + ex_nums[i] + '_grid_pi'
        model = load_model(operator_path, shape)
        model.set_init(in_nn)

        # model step
        for k in range(t_start, nt):
            model.step(f_nn[k])
        model.combine_data(data_ts)
        model.compute_error(data_sps)
        model.save_logs(logs[ex_nums[i]])
        model.plot(ax, t_nn, label[i])

    # ax1.set_ylim(2.5, 3.5)
    # ax2.set_ylim(-1.5, 1.5)
    ax[2].set_yscale('log')
    ax[2].set_ylim(1e-4, 1e1)
    ax[3].set_yscale('log')
    ax[3].set_ylim(1e-3, 1e1)

    plt.savefig(f'logs/coef_phase1_scale_{scale}.jpg')
    torch.save(logs, logs_path)
