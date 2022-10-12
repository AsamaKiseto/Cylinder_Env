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
        # ax[i].set_ylabel(f'loss{i+1}', fontsize=15)
        ax[i].set_ylim(1e-4, 1)

    ax[0].set_title("loss plot", fontsize=10)
    ax[0].set_ylabel("pred loss", fontsize=10)
    ax[1].set_ylabel("phys loss of obs", fontsize=10)
    ax[2].set_ylabel("phys loss of pred", fontsize=10)
    ax[2].set_xlabel("epochs", fontsize=10)

    for k in range(len(log_list)):
        loss = torch.load('logs/data/loss_log_' + log_list[k])
        for i in range(3):
            ax[i].plot(loss[i], label=log_list[k])
            ax[i].legend()
    
    plt.savefig(f'logs/loss_plot_{fig_name}.jpg')


def test_plot(t_nn, log_list, scale_k, ex_name = 'fb_0.0', fig_name = 'test'):
    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim(5e-4, 5)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=15)
    ax[0].set_ylabel("One-step data loss", fontsize=10)
    ax[1].set_ylabel("Cumul data loss", fontsize=10)
    ax[2].set_ylabel("phys loss of obs", fontsize=10)
    ax[3].set_ylabel("phys loss of pred", fontsize=10)
    ax[3].set_xlabel("t", fontsize=10)

    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data/phase1_test_{log_list[k]}_{ex_name}')
        error_1step, Lpde_obs, Lpde_pred, error_cul, _, _, _, _, _ = calMean(data_list)
        error_1step_v, Lpde_obs_v, Lpde_pred_v, error_cul_v, _, _, _, _, _ = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[2].plot(t_nn, Lpde_obs[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[3].plot(t_nn, Lpde_pred[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            
            ax[0].fill_between(t_nn, error_1step_v[scale_k[j]][0], error_1step_v[scale_k[j]][1], alpha=0.2)
            ax[1].fill_between(t_nn, error_cul_v[scale_k[j]][0], error_cul_v[scale_k[j]][1], alpha=0.2)
            ax[2].fill_between(t_nn, Lpde_obs_v[scale_k[j]][0], Lpde_obs_v[scale_k[j]][1], alpha=0.2)
            ax[3].fill_between(t_nn, Lpde_pred_v[scale_k[j]][0], Lpde_pred_v[scale_k[j]][1], alpha=0.2)
            
            ax[0].legend()
            plt.savefig(f'logs/pics/phase1_{fig_name}_scale_{scale[scale_k[j]]}.jpg')
    
    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=500)
    ax = ax.flatten()
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim(5e-4, 5)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=15)
    ax[0].set_ylabel(r'One-step $C_D$ loss', fontsize=10)
    ax[1].set_ylabel(r'One-step $C_L$ loss', fontsize=10)
    ax[2].set_ylabel(r'Cul $C_D$ loss', fontsize=10)
    ax[3].set_ylabel(r'Cul $C_L$ loss', fontsize=10)
    ax[3].set_xlabel("t", fontsize=10)
    
    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data/phase1_test_{log_list[k]}_{ex_name}')
        _, _, _, _, _, error_Cd_1step, error_Cl_1step, error_Cd_cul, error_Cl_cul = calMean(data_list)
        _, _, _, _, _, error_Cd_1step_v, error_Cl_1step_v, error_Cd_cul_v, error_Cl_cul_v = calMean(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_Cd_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_Cl_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[2].plot(t_nn, error_Cd_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[3].plot(t_nn, error_Cl_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            
            ax[0].fill_between(t_nn, error_Cd_1step_v[scale_k[j]][0], error_Cd_1step_v[scale_k[j]][1], alpha=0.2)
            ax[1].fill_between(t_nn, error_Cl_1step_v[scale_k[j]][0], error_Cl_1step_v[scale_k[j]][1], alpha=0.2)
            ax[2].fill_between(t_nn, error_Cd_cul_v[scale_k[j]][0], error_Cd_cul_v[scale_k[j]][1], alpha=0.2)
            ax[3].fill_between(t_nn, error_Cl_cul_v[scale_k[j]][0], error_Cl_cul_v[scale_k[j]][1], alpha=0.2)
            
            ax[0].legend()
            
            plt.savefig(f'logs/pics/phase1_coef_{fig_name}_scale_{scale[scale_k[j]]}.jpg')