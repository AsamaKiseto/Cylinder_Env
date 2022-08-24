import torch
import matplotlib.pyplot as plt 
import argparse
from matplotlib import colors

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    parser.add_argument('-n', '--ex_nums', nargs='+', type=int, help='experiment number')
    parser.add_argument('--y_min', default=1e-3, type=float, help='y axis limits')
    parser.add_argument('--y_max', default=1, type=float, help='y axis limits')
    return parser.parse_args(argv)

def add_plots(ax, logs, label):
    loss1, loss2, loss3, loss4 = logs['test_loss_trans'], logs['test_loss_u_t_rec'], logs['test_loss_f_t_rec'], logs['test_loss_trans_latent']
    for i in range(4):
        exec(f'ax[{i}].plot(loss{i+1}, label="Ex"+str(label))')
        exec(f'ax[{i}].legend()')

if __name__ == '__main__':
    # argparser
    args = get_args()
    ex_nums = args.ex_nums
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
    N = len(ex_nums)
    print(ex_nums)
    _, logs_base = torch.load(f"logs/phase1_ex{ex_nums[0]}_norm")
    loss1, loss2, loss3, loss4 = logs_base['test_loss_trans'], logs_base['test_loss_u_t_rec'], logs_base['test_loss_f_t_rec'], logs_base['test_loss_trans_latent']
    for i in range(4):
        exec(f'ax[{i}].plot(loss{i+1}, color="black", label="Ex"+str({ex_nums[0]}))')
        exec(f'ax[{i}].legend()')

    for i in range(1, N):
        exec(f'_, logs_ex{ex_nums[i]} = torch.load("logs/phase1_ex{ex_nums[i]}_norm")')
        exec(f'add_plots(ax, logs_ex{ex_nums[i]}, label={ex_nums[i]})')

    plt.savefig(f'logs/loss_plot_norm_ex_{ex_nums}.jpg')