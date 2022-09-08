import torch
import numpy as np
import matplotlib.pyplot as plt 
import argparse
from matplotlib import colors

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    parser.add_argument('-tl', '--t_length', default=80, type=int, help='y axis limits')
    parser.add_argument('--y_min', default=1e-3, type=float, help='y axis limits')
    parser.add_argument('--y_max', default=1, type=float, help='y axis limits')
    return parser.parse_args(argv)

def add_plots(ax, logs, tl, label):
    scale, Cd_var, Cl_var, obs_var, Lpde_nn = logs['scale'], logs['Cd_var'], logs['Cl_var'], logs['obs_var'], logs['Lpde_nn']
    loss1, loss2, loss3, loss4 = np.asarray(Cd_var)[:, :tl].mean(-1), np.asarray(Cl_var)[:, :tl].mean(-1), \
                                 np.asarray(obs_var)[:, :tl].mean(-1), np.asarray(Lpde_nn)[:, :tl].mean(-1)
    scale = np.asarray(scale)
    for i in range(4):
        exec(f'ax[{i}].plot(scale, loss{i+1}, label=label)')
        exec(f'ax[{i}].legend()')

if __name__ == '__main__':
    # argparser
    args = get_args()
    y_min = args.y_min
    y_max = args.y_max
    tl = args.t_length
    
    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    # plt.figure(figsize=(15, 12))
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_yscale('log')
        ax[i].set_ylim(y_min, y_max)

    ax[0].set_title(r"$error/loss in different scales$", fontsize=15)
    ax[0].set_ylabel(r"$C_d$", fontsize=15)
    ax[1].set_ylabel(r"$C_l$", fontsize=15)
    ax[2].set_ylabel(r"$state$", fontsize=15)
    ax[3].set_ylabel(r"$L_{pde}$", fontsize=15)
    ax[3].set_xlabel(r"$scale$", fontsize=15)

    # load logs
    ex_nums = ['ex0_big', 'ex3_big', 'ex3_big_nomod']
    label = ['base(without pde loss)', 'with modify', 'with without modify']
    N = len(ex_nums)
    print(ex_nums)

    logs_base = torch.load(f"logs/phase1_env_logs_{ex_nums[i]}")
    # Cd_var, Cl_var, obs_var, Lpde_nn
    scale, Cd_var, Cl_var, obs_var, Lpde_nn = logs_base['scale'], logs_base['Cd_var'], logs_base['Cl_var'], logs_base['obs_var'], logs_base['Lpde_nn']
    loss1, loss2, loss3, loss4 = np.asarray(Cd_var)[:, :tl].mean(-1), np.asarray(Cl_var)[:, :tl].mean(-1), \
                                 np.asarray(obs_var)[:, :tl].mean(-1), np.asarray(Lpde_nn)[:, :tl].mean(-1)
    scale = np.asarray(scale)
    for i in range(4):
        exec(f'ax[{i}].plot(scale, loss{i+1}, color="black", label={ex_nums[0]})')
        exec(f'ax[{i}].legend()')

    for i in range(N):
        exec(f'logs_ex{ex_nums[i]} = torch.load("logs/phase1_env_logs_{ex_nums[i]}")')
        exec(f'add_plots(ax, logs_ex{ex_nums[i]}, tl, label={label[i]})')

    plt.savefig(f'logs/loss_genrlty_{tl}.jpg')