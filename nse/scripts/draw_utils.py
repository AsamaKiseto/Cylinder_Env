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
        loss = torch.load('logs/data/' + log_list[k])
        for i in range(3):
            ax[i].plot(loss[i], label=log_list[k])
            ax[i].legend()
    
    plt.savefig(f'logs/loss_plot_{fig_name}.jpg')


def test_plot(log_list, t_nn, scale_k, fig_name = 'test'):
    scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim()
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=15)
    ax[0].set_ylabel("One-step data loss", fontsize=15)
    ax[1].set_ylabel("Cumul data loss", fontsize=15)
    ax[2].set_ylabel("phys loss of obs", fontsize=15)
    ax[3].set_ylabel("phys loss of pred", fontsize=15)
    ax[3].set_xlabel("t", fontsize=15)

    for k in range(len(log_list)):
        data_list = torch.load('logs/data/phase1_test_' + log_list[k])
        error_1step, Lpde_obs, Lpde_pred, error_cul, Lpde_pred_cul = calMean(data_list)
        ax[0].plot(t_nn, error_1step[scale_k], label=log_list[k])
        ax[1].plot(t_nn, error_cul[scale_k], label=log_list[k])
        ax[2].plot(t_nn, Lpde_obs[scale_k], label=log_list[k])
        ax[3].plot(t_nn, Lpde_pred[scale_k], label=log_list[k])
        for i in range(4):
            ax[i].legend()
    
    plt.savefig(f'logs/pics/coef_phase1_{fig_name}_scale_{scale[scale_k]}.jpg')