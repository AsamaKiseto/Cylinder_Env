import numpy as np
import matplotlib.pyplot as plt 
import torch

from scripts.models import *
from scripts.nse_model import *
from scripts.utils import *
from scripts.draw_utils import *

dt = 0.01
tg = 5
nt = 80
t_start = 5
t_nn = (np.arange(nt) + 1) * 0.01 * tg
t = (np.arange(nt * tg) + 1) * 0.01 

ex_nums = ['ex0', 'ex1', 'ex4']
label = ['baseline', '2-step', '1-step']
ex_nums = ['ex0', 'ex1']
label = ['baseline', 'phys-included']
ex_nums = ['ex0']
n_model = len(ex_nums)

# fig setting
fig2, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
ax = ax.flatten()
for i in range(4):
    ax[i] = plt.subplot2grid((4, 2), (i, 0), colspan=2)
    ax[i].grid(True, lw=0.4, ls="--", c=".50")
    ax[i].set_xlim(0, nt * tg * dt)
    
ax[0].set_title("Error/Loss in Different Scales", fontsize=15)
ax[0].set_ylabel("One-step data loss", fontsize=15)
ax[1].set_ylabel("Cumul data loss", fontsize=15)
ax[2].set_ylabel("phys loss of obs", fontsize=15)
ax[3].set_ylabel("phys loss of pred", fontsize=15)
ax[3].set_xlabel("t", fontsize=15)

if __name__ == '__main__':
    # load test data
    data_path = 'data/test_data/nse_data_reg_scale...'
    data_path = 'data/nse_data_reg'
    data = LoadData(data_path)
    data.split(1, 5)
    obs, Cd, Cl, ctr = data.get_data()
    data_norm = data.norm()
    in_nn = obs[:, t_start].unsqueeze(1)

    N0, nt, nx, ny = data.get_params()
    print(N0, nt, nx, ny)
    shape = [nx, ny]

    for k in range(n_model):
        operator_path = 'logs/phase1_' + ex_nums[k] + '_grid_pi'
        model = LoadModel(operator_path, shape)
        
        error_1step, Lpde_obs, Lpde_pred = torch.zeros(N0, nt), torch.zeros(N0, nt), torch.zeros(N0, nt) 
        error_cul, Lpde_pred_cul = torch.zeros(N0, nt), torch.zeros(N0, nt)
        
        for n in range(N0):
            model.set_init(in_nn[n])
            error_cul[n], Lpde_pred_cul[n] = model.process(obs[n], Cd[n], Cl[n], ctr[n], t_start)
            print(f'# {n} Lpde_nn: {Lpde_pred_cul[n]}')
            error_1step[n], Lpde_obs[n], Lpde_pred[n] = model.cal_1step(obs[n], Cd[n], Cl[n], ctr[n])
            print(f'# {n} error_1step: {error_1step[n]}')
            # print(f'Lpde_obs: {Lpde_obs[n]}')
            # print(f'Lpde_pred: {Lpde_pred[n]}')

        ax[0].plot(t_nn, error_1step.mean(0), label=label[k])
        ax[1].plot(t_nn, error_cul.mean(0), label=label[k])
        ax[2].plot(t_nn, Lpde_obs.mean(0), label=label[k])
        ax[3].plot(t_nn, Lpde_pred.mean(0), label=label[k])

        for i in range(4):
            ax[i].legend()

    plt.savefig('logs/pics/coef_phase1_test.jpg')