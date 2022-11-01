import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 
from timeit import default_timer

from scripts.nets import *
from scripts.utils import *
from scripts.draw_utils import *

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    
    parser.add_argument('-lf', '--log_file', default='test', type=str, help='log file name')
    parser.add_argument('-nt', '--nt', default=10, type=int, help='nums of timestamps')
    parser.add_argument('-tg', '--tg', default=5, type=int, help='gap of timestamps')
    parser.add_argument('-s', '--scale', default=0.5, type=float, help='random scale')
    parser.add_argument('-f', '--f_base', default=0.142857, type=float, help='base f')
    parser.add_argument('-ts', '--t_start', default=5, type=int, help='data number')

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

print(env.params)

class load_model():
    def __init__(self, operator_path, shape):
        # mosel params setting
        print(operator_path)
        state_dict_pred, state_dict_phys, logs_model = torch.load(operator_path)
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

        self.pred_model = FNO_ensemble(model_params)
        self.pred_model.load_state_dict(state_dict_pred)
        self.pred_model.eval()
        self.phys_model = state_mo(model_params)
        self.phys_model.load_state_dict(state_dict_phys)
        self.phys_model.eval()

        self.logs = dict()
        self.logs['obs_nn']=[]
        self.logs['Cd_nn']=[]
        self.logs['Cl_nn']=[]
        self.logs['Lpde_nn']=[]
        self.logs['Lpde_obs']=[]
    
    def cul_1step(self, obs, ctr):
        nt, nx, ny = obs.shape[0] - 1, obs.shape[1], obs.shape[2]
        out_nn = np.zeros((nt, nx, ny, 3))
        Cd_nn, Cl_nn = np.zeros(nt), np.zeros(nt)
        for k in range(nt):
            pred, _, _, _ = self.pred_model(torch.Tensor(obs_sps[k]).unsqueeze(0), ctr[k].reshape(1))
            out_nn[k] = pred[:, :, :, :3].detach().numpy()
            Cd_nn[k] = torch.mean(pred[:, :, :, -2]).detach().numpy()
            Cl_nn[k] = torch.mean(pred[:, :, :, -1]).detach().numpy()
        
        Cd_mean, Cd_var = self.data_norm['Cd']
        Cl_mean, Cl_var = self.data_norm['Cl']
        Cd_nn = Cd_nn * Cd_var.item() + Cd_mean.item()
        Cl_nn = Cl_nn * Cl_var.item() + Cl_mean.item()
        return out_nn, Cd_nn, Cl_nn

    def step(self, ctr_nn):
        pred, _, _, _ = self.pred_model(self.in_nn, ctr_nn.reshape(1))
        out_nn = pred[:, :, :, :3]
        mod = self.phys_model(self.in_nn, ctr_nn.reshape(1), out_nn)
        Lpde_nn = ((Lpde(out_nn, self.in_nn, self.dt) + mod) ** 2).mean()
        print(f'Lpde_nn: {Lpde_nn}')
        Cd_nn = torch.mean(pred[:, :, :, -2])
        Cl_nn = torch.mean(pred[:, :, :, -1])

        Cd_mean, Cd_var = self.data_norm['Cd']
        Cl_mean, Cl_var = self.data_norm['Cl']
        Cd_nn = Cd_nn * Cd_var + Cd_mean
        Cl_nn = Cl_nn * Cl_var + Cl_mean

        self.in_nn = out_nn
        self.logs['Cd_nn'].append(Cd_nn.detach().numpy())
        self.logs['Cl_nn'].append(Cl_nn.detach().numpy())
        self.logs['obs_nn'].append(out_nn.squeeze().detach().numpy())
        self.logs['Lpde_nn'].append(Lpde_nn.detach().numpy())
    
    def cal_Lpde_obs(self, obs_in, ctr, obs_out):
        mod = self.phys_model(obs_in, ctr.reshape(1), obs_out)
        Lpde_obs = ((Lpde(obs_out, obs_in, self.dt) + mod) ** 2).mean().detach().numpy()
        self.logs['Lpde_obs'].append(Lpde_obs)
    
    def set_init(self, state_nn):
        self.in_nn = state_nn
        self.in_obs = state_nn

    def combine_data(self, data_ts, t_start):
        obs_ts, Cd_ts, Cl_ts = data_ts
        Lpde_ts = np.zeros(t_start)
        self.obs_nn = np.concatenate((obs_ts, np.asarray(self.logs['obs_nn'])), 0)
        self.Cd_nn = np.concatenate((Cd_ts, np.asarray(self.logs['Cd_nn'])), 0)
        self.Cl_nn = np.concatenate((Cl_ts, np.asarray(self.logs['Cl_nn'])), 0)
        self.Lpde_nn = np.concatenate((Lpde_ts, np.asarray(self.logs['Lpde_nn'])), 0)
        self.Lpde_obs = np.asarray(self.logs['Lpde_obs'])

    def compute_error(self, data_sps):
        obs_sps, Cd_sps, Cl_sps = data_sps
        nt = obs_sps.shape[0] - 1
        Cd_var = Cd_sps - self.Cd_nn
        Cl_var = Cl_sps - self.Cl_nn
        obs_var = obs_sps[1:] - self.obs_nn
        self.Cd_var = (Cd_var.reshape(nt, -1)**2).mean(1)
        self.Cl_var = (Cl_var.reshape(nt, -1)**2).mean(1)
        self.obs_var = (obs_var.reshape(nt, -1)**2).mean(1)

    def plot(self, ax, t_nn, t_start, label=None):
        ax[0].plot(t_nn[t_start:], self.obs_var[t_start:], label=f'{label}')
        ax[1].plot(t_nn[t_start:], self.Cd_var[t_start:], label=f'{label}')
        ax[2].plot(t_nn[t_start:], self.Cl_var[t_start:], label=f'{label}')
        ax[3].plot(t_nn[t_start:], self.Lpde_obs[t_start:], label=f'{label}')
        ax[4].plot(t_nn[t_start:], self.Lpde_nn[t_start:], label=f'{label}')
        for i in range(5):
            ax[i].legend()
    
    def save_logs(self, logs):
        logs['obs_nn'].append(self.obs_nn)
        logs['Cd_nn'].append(self.Cd_nn)
        logs['Cl_nn'].append(self.Cl_nn)
        logs['Lpde_nn'].append(self.Lpde_nn)
        logs['Cd_var'].append(self.Cd_var)
        logs['Cl_var'].append(self.Cl_var)
        logs['obs_var'].append(self.obs_var)
        logs['Lpde_obs'].append(self.Lpde_obs)

if __name__ == '__main__':
    # argparser
    args = get_args()

    # path
    logs_path = f'logs/phase1_env_logs_{args.log_file}'

    # ex_nums = ['ex0', 'ex1_3', 'ex4_3']
    # label = ['baseline', '2-step', '1-step']
    ex_nums = ['ex0', 'ex1_3', 'ex1_extra']
    label = ['baseline', 'phys-tog', 'phys-dev']
    label = ['baseline', 'phys-include', 'extra_train']
    
    # label = [ 'with_modify']
    n_model = len(ex_nums)

    # logs
    if not os.path.isfile(logs_path):
        logs=dict()

        logs['ex_nums'] = ex_nums
        logs['label'] = label
        logs['scale']=[]
        logs['t_start']=[]
        logs['ctr'] = []
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
            logs[ex_nums[i]]['Lpde_obs']=[]
            logs[ex_nums[i]]['error_1step']=[]

    else:
        logs = torch.load(logs_path)

    # data param
    nx, ny = env.params['dimx'], env.params['dimy']
    shape = [nx, ny]
    dt = env.params['dt']

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
    f = scale * (np.random.rand(nt) - 0.5) + args.f_base
    logs['ctr'].append(f)
    # f = scale * (np.random.rand(nt) - 0.5) + 1
    # f = np.ones(nt) * 0.111111111
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
    
    obs_sps, Cd_sps, Cl_sps = obs[::tg][...,2:], Cd[tg-1::tg], Cl[tg-1::tg]
    obs_ts, Cd_ts, Cl_ts = obs_sps[1:t_start+1], Cd_sps[:t_start], Cl_sps[:t_start]
    data_ts = [obs_ts, Cd_ts, Cl_ts]
    data_sps = [obs_sps, Cd_sps, Cl_sps]

    logs['obs'].append(obs)
    logs['Cd'].append(Cd)
    logs['Cl'].append(Cl)

    # fig setting
    fig_num = 5
    fig, ax = plt.subplots(nrows=fig_num, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_xlim(0, nt * tg * dt)
        ax[i].set_yscale('log')
        
    ax[0].set_title("error/loss in different scales", fontsize=15)
    ax[0].set_ylabel("state error", fontsize=15)
    ax[1].set_ylabel("Cd error", fontsize=15)
    ax[2].set_ylabel("Cl error", fontsize=15)
    ax[3].set_ylabel("phys loss of obs", fontsize=15)
    ax[4].set_ylabel("phys loss of pred", fontsize=15)
    ax[4].set_xlabel("t", fontsize=15)

    ax[0].set_ylim(1e-4, 1)
    ax[1].set_ylim(1e-4, 1)
    ax[2].set_ylim(1e-4, 1)
    ax[3].set_ylim(1e-3, 1e2)
    ax[4].set_ylim(1e-3, 1e2)
    
    # mosel setting
    for i in range(n_model):
        operator_path = 'logs/phase1_' + ex_nums[i] + '_grid_pi'
        model = load_model(operator_path, shape)
        model.set_init(in_nn)
        error_1step = np.zeros(nt)
        out_nn = np.zeros((nt, nx, ny, 3))
        Cd_nn, Cl_nn = np.zeros(nt), np.zeros(nt)
        
        # one step pred
        # out_nn, Cd_nn, Cl_nn = model.cul_1step(obs_sps, f_nn)
        # # print(Cd_nn, Cd_sps)
        # # print(Cl_nn, Cl_sps)
        # error_1step = ((out_nn - obs_sps[1:])**2).reshape(nt, -1).mean(1) + \
        #               ((Cd_nn - Cd_sps)**2).reshape(nt, -1).mean(1) + \
        #               ((Cl_nn - Cl_sps)**2).reshape(nt, -1).mean(1)
        # # print(error_1step)
        # logs[ex_nums[i]]['error_1step'].append(error_1step)
        # ax[0].plot(t_nn[t_start:], error_1step[t_start:], label=f'{label[i]}')
        # ax[0].legend()

        # model step
        for k in range(t_start):
            model.cal_Lpde_obs(torch.Tensor(obs_sps[k]).unsqueeze(0), f_nn[k], torch.Tensor(obs_sps[k+1]).unsqueeze(0))
        for k in range(t_start, nt):
            model.step(f_nn[k])
            model.cal_Lpde_obs(torch.Tensor(obs_sps[k]).unsqueeze(0), f_nn[k], torch.Tensor(obs_sps[k+1]).unsqueeze(0))
        model.combine_data(data_ts, t_start)
        model.compute_error(data_sps)
        model.save_logs(logs[ex_nums[i]])
        model.plot(ax, t_nn, t_start, label[i])

    ax[0].set_yscale('log')
    ax[0].set_ylim(1e-5, 1)
    ax[1].set_yscale('log')
    ax[1].set_ylim(1e-4, 1e1)
    ax[2].set_yscale('log')
    ax[2].set_ylim(1e-3, 1e2)
    ax[3].set_yscale('log')
    ax[3].set_ylim(1e-3, 1e2)

    plt.savefig(f'logs/pics/coef_phase1_{args.log_file}_{scale}.jpg')
    torch.save(logs, logs_path)