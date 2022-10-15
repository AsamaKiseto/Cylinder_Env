from scripts.utils import *
import matplotlib.pyplot as plt 
import torch

# draw loss plot
def loss_plot(log_list, fig_name = 'test'):
    # fig setting
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,12), dpi=500)
    ax = ax.flatten()
    # plt.figure(figsize=(15, 12))
    for i in range(3):
        ax[i] = plt.subplot2grid((3, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_yscale('log')
        # ax[i].set_ylabel(f'loss{i+1}', fontsize=20)
        ax[i].set_ylim(1e-4, 1)

    ax[0].set_title("loss plot", fontsize=20)
    ax[0].set_ylabel("pred loss", fontsize=20)
    ax[1].set_ylabel("phys loss of obs", fontsize=20)
    ax[2].set_ylabel("phys loss of pred", fontsize=20)
    ax[2].set_xlabel("epochs", fontsize=20)

    for k in range(len(log_list)):
        loss = torch.load('logs/data/loss_log_' + log_list[k])
        for i in range(3):
            ax[i].plot(loss[i], label=log_list[k])
            ax[i].legend()
    
    plt.savefig(f'logs/loss_plot_{fig_name}.jpg')


def test_plot(t_nn, log_list, scale_k, ex_name = 'fb_0.0', fig_name = 'test'):
    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # state error fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_ylim(1e-4, 1)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel("One-step data error", fontsize=20)
    ax[1].set_ylabel("Cumul data error", fontsize=20)
    ax[fig_num - 1].set_xlabel("t", fontsize=20)

    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
        error_1step, error_cul, _, _, _, _ = calMean(data_list)
        error_1step_v, error_cul_v, _, _, _, _ = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')

            # ax[0].fill_between(t_nn, error_1step_v[0][scale_k[j]], error_1step_v[1][scale_k[j]], alpha=0.2)
            # ax[1].fill_between(t_nn, error_cul_v[0][scale_k[j]], error_cul_v[1][scale_k[j]], alpha=0.2)
            
            ax[0].legend()
            
    plt.savefig(f'logs/pics/error/phase1_{fig_name}_{ex_name}.jpg')
    
    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_yscale('log')
        ax[i].set_ylim(0, 0.5)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel(r'Cul $C_D$ error', fontsize=20)
    ax[1].set_ylabel(r'Cul $C_L$ error', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)
    
    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
        _, _, _, _, error_Cd_cul, error_Cl_cul = calMean(data_list)
        _, _, _, _, error_Cd_cul_v, error_Cl_cul_v = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_Cd_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_Cl_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            
            # ax[0].fill_between(t_nn, error_Cd_cul_v[0][scale_k[j]], error_Cd_cul_v[1][scale_k[j]], alpha=0.2)
            # ax[1].fill_between(t_nn, error_Cl_cul_v[0][scale_k[j]], error_Cl_cul_v[1][scale_k[j]], alpha=0.2)
            
            ax[0].legend()
            
    plt.savefig(f'logs/pics/error/phase1_coef_cul_{fig_name}_{ex_name}.jpg')
    
    # 1 step coef fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e-4, 1e-1)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel(r'One-step $C_D$ error', fontsize=20)
    ax[1].set_ylabel(r'One-step $C_L$ error', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)
    
    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
        _, _, error_Cd_1step, error_Cl_1step, _, _ = calMean(data_list)
        _, _, error_Cd_1step_v, error_Cl_1step_v, _, _ = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_Cd_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_Cl_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            
            # ax[0].fill_between(t_nn, error_Cd_1step_v[0][scale_k[j]], error_Cd_1step_v[1][scale_k[j]], alpha=0.2)
            # ax[1].fill_between(t_nn, error_Cl_1step_v[0][scale_k[j]], error_Cl_1step_v[1][scale_k[j]], alpha=0.2)
            
            ax[0].legend()
            
    plt.savefig(f'logs/pics/error/phase1_coef_1step_{fig_name}_{ex_name}.jpg')
    
def coef_plot(t_nn, scale_k, data, fig_name):
    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    Cd_mean, Cl_mean = calMean(data)
    Cd_var, Cl_var = calVar(data)

    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_yscale('log')
        # ax[i].set_ylim(5e-4, 5)
        
    ax[0].set_title("Test Obs", fontsize=20)
    ax[0].set_ylabel(r'$C_D$', fontsize=20)
    ax[1].set_ylabel(r'$C_L$', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)

    for j in range(len(scale_k)):
        ax[0].plot(t_nn, Cd_mean[scale_k[j]], label=scale[scale_k[j]])
        ax[1].plot(t_nn, Cl_mean[scale_k[j]], label=scale[scale_k[j]])
        
        ax[0].fill_between(t_nn, Cd_var[0][scale_k[j]], Cd_var[1][scale_k[j]], alpha=0.2)
        ax[1].fill_between(t_nn, Cl_var[0][scale_k[j]], Cl_var[1][scale_k[j]], alpha=0.2)

    plt.savefig(f'logs/obs_coef_{fig_name}.jpg')
    
def coef_plot1(t_nn, data, fig_name):
    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    Cd, Cl = data
    Cd_mean, Cl_mean = Cd.mean(0), Cl.mean(0)
    Cd_var, Cl_var = [Cd.min(0), Cd.max(0)], [Cl.min(0), Cl.max(0)]

    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_yscale('log')
        # ax[i].set_ylim(5e-4, 5)
        
    ax[0].set_title("Test Obs", fontsize=20)
    ax[0].set_ylabel(r'$C_D$', fontsize=20)
    ax[1].set_ylabel(r'$C_L$', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)

    for i in range(Cd.shape[0]):
        ax[0].plot(t_nn, Cd[i])
        ax[1].plot(t_nn, Cl[i])

    plt.savefig(f'logs/obs_coef_{fig_name}.jpg')