import torch
import matplotlib.pyplot as plt 
import argparse
from matplotlib import colors

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    # parser.add_argument('-n', '--ex_nums', nargs='+', type=int, help='experiment number')
    parser.add_argument('--y_min', default=1e-3, type=float, help='y axis limits')
    parser.add_argument('--y_max', default=1, type=float, help='y axis limits')
    return parser.parse_args(argv)

def add_plots(ax, logs, label):
    loss1, loss2, loss3, loss4, loss5 = logs['test_loss_trans'], logs['test_loss_u_t_rec'], \
                                        logs['test_loss_f_t_rec'], logs['test_loss_trans_latent'], logs['test_loss_pde']
    for i in range(5):
        exec(f'ax[{i}].plot(loss{i+1}, label=label)')
        exec(f'ax[{i}].legend()')

if __name__ == '__main__':
    # argparser
    args = get_args()
    y_min = args.y_min
    y_max = args.y_max
    
    # fig setting
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    # plt.figure(figsize=(15, 12))
    for i in range(4):
        ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_yscale('log')
        ax[i].set_ylabel(f'loss{i+1}', fontsize=15)
        ax[i].set_ylim(y_min, y_max)

    # load logs
    # ex_nums = ['ex0', 'ex3', 'ex3_nomod']
    ex_nums = ['ex0_big', 'ex3_big', 'ex3_big_nomod']
    label = ['base', 'modify', 'without modify']
    N = len(ex_nums)
    print(ex_nums)
    _, logs_base = torch.load(f"logs/phase1_{ex_nums[0]}_grid_pi")
    logs_base = logs_base['logs']
    loss1, loss2, loss3, loss4, loss5 = logs_base['test_loss_trans'], logs_base['test_loss_u_t_rec'], \
                                        logs_base['test_loss_f_t_rec'], logs_base['test_loss_trans_latent'], logs_base['test_loss_pde']
    for i in range(5):
        exec(f'ax[{i}].plot(loss{i+1}, color="black", label={label[0]})')
        exec(f'ax[{i}].legend()')

    for i in range(1, N):
        exec(f'_, logs_ex{ex_nums[i]} = torch.load("logs/phase1_{ex_nums[i]}_grid_pi")')
        exec(f'add_plots(ax, logs_ex{ex_nums[i]}["logs"], label={label[i]})')

    plt.savefig('logs/loss_plot.jpg')