import numpy as np
import matplotlib.pyplot as plt 
import torch

# draw loss plot
def add_loss_plots(ax, logs, label):
    loss1, loss2, loss3, loss4, loss5 = logs['test_loss_trans'], logs['test_loss_u_t_rec'], \
                                        logs['test_loss_f_t_rec'], logs['test_loss_trans_latent'], logs['test_loss_pde']
    for i in range(5):
        exec(f'ax[{i}].plot(loss{i+1}, label=label)')
        exec(f'ax[{i}].legend()')

def draw_loss_plot(ex_nums, label):
    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    # plt.figure(figsize=(15, 12))
    for i in range(5):
        ax[i] = plt.subplot2grid((5, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_yscale('log')
        # ax[i].set_ylabel(f'loss{i+1}', fontsize=15)
        ax[i].set_ylim(1e-3, 1)
    
    ax[0].set_ylabel("pred loss", fontsize=10)
    ax[1].set_ylabel("recon loss of state", fontsize=10)
    ax[2].set_ylabel("recon loss of", fontsize=10)
    ax[3].set_ylabel("latent loss", fontsize=10)
    ax[4].set_ylabel("physical loss", fontsize=10)

    # load logs
    N = len(ex_nums)
    print(ex_nums)
    _, logs_base = torch.load(f"logs/phase1_{ex_nums[0]}_grid_pi")
    logs_base = logs_base['logs']
    loss1, loss2, loss3, loss4, loss5 = logs_base['test_loss_trans'], logs_base['test_loss_u_t_rec'], \
                                        logs_base['test_loss_f_t_rec'], logs_base['test_loss_trans_latent'], logs_base['test_loss_pde']
    for i in range(5):
        exec(f'ax[{i}].plot(loss{i+1}, color="black", label="{label[0]}")')
        exec(f'ax[{i}].legend()')

    for i in range(1, N):
        exec(f'_, logs_ex{ex_nums[i]} = torch.load("logs/phase1_{ex_nums[i]}_grid_pi")')
        exec(f'add_loss_plots(ax, logs_ex{ex_nums[i]}["logs"], label="{label[i]}")')

    plt.savefig('logs/loss_plot.jpg')

# draw generality of scale
def add_generality_plots(ax, scale, logs, tl, label):
    Cd_var, Cl_var, obs_var, Lpde_nn = logs['Cd_var'], logs['Cl_var'], logs['obs_var'], logs['Lpde_nn']
    loss1, loss2, loss3, loss4 = np.asarray(Cd_var)[:, :tl].mean(-1), np.asarray(Cl_var)[:, :tl].mean(-1), \
                                 np.asarray(obs_var)[:, :tl].mean(-1), np.asarray(Lpde_nn)[:, :tl].mean(-1)
    for i in range(4):
        exec(f'ax[{i}].plot(scale, loss{i+1}, label=label)')
        exec(f'ax[{i}].legend()')

def draw_generality(logs, ex_nums, label, tl):
    
    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    # plt.figure(figsize=(15, 12))
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_yscale('log')
        # ax[i].set_ylim(y_min, y_max)

    ax[0].set_title("error/loss in different scales", fontsize=15)
    ax[0].set_ylabel(r"$C_d$", fontsize=15)
    ax[1].set_ylabel(r"$C_l$", fontsize=15)
    ax[2].set_ylabel(r"$state$", fontsize=15)
    ax[3].set_ylabel(r"$L_{pde}$", fontsize=15)
    ax[3].set_xlabel(r"$scale$", fontsize=15)

    # load logs
    
    N = len(ex_nums)
    print(ex_nums)

    logs_base = logs[f'{ex_nums[0]}']
    scale = logs['scale']
    # Cd_var, Cl_var, obs_var, Lpde_nn
    Cd_var, Cl_var, obs_var, Lpde_nn = logs_base['Cd_var'], logs_base['Cl_var'], logs_base['obs_var'], logs_base['Lpde_nn']
    loss1, loss2, loss3, loss4 = np.asarray(Cd_var)[:, :tl].mean(-1), np.asarray(Cl_var)[:, :tl].mean(-1), \
                                 np.asarray(obs_var)[:, :tl].mean(-1), np.asarray(Lpde_nn)[:, :tl].mean(-1)
    scale = np.asarray(scale)
    for i in range(4):
        exec(f'ax[{i}].plot(scale, loss{i+1}, color="black", label="{label[0]}")')
        exec(f'ax[{i}].legend()')

    for i in range(1, N):
        exec(f'logs_{ex_nums[i]} = logs["{ex_nums[i]}"]')
        exec(f'add_generality_plots(ax, scale, logs_{ex_nums[i]}, tl, label="{label[i]}")')

    plt.savefig(f'logs/loss_genrlty_{tl}.jpg')