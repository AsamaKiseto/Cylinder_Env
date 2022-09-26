import numpy as np
import matplotlib.pyplot as plt 
import torch

from scripts.models import *
from scripts.nse_model import *
from scripts.utils import *
from scripts.draw_utils import *

tg = 5
nt = 80
t_start = 5
t_nn = (np.arange(nt) + 1) * 0.01 * tg
t = (np.arange(nt * tg) + 1) * 0.01 

ex_nums = ['ex0', 'ex1', 'ex4']
label = ['baseline', '2-step', '1-step']
# ex_nums = ['ex0', 'ex1']
# label = ['baseline', 'phys-included']
n_model = len(ex_nums)

# fig setting
fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(15,12), dpi=1000)
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

# load test data
data_path = 'data/test_data/nse_data_reg_scale...'
data = LoadData(data_path)
data.split(1, 5)
obs, Cd, Cl, ctr = data.get_data()
data_norm = data.norm()
in_nn = obs[:, t_start]

N0, nt, nx, ny = data.get_params()
shape = [nx, ny]

obs_nn, Cd_nn, Cl_nn = torch.zeros(N0, nt, nx, ny, 3), torch.zeros(N0, nt), torch.zeros(N0, nt)
for i in range(n_model):
    operator_path = 'logs/phase1_' + ex_nums[i] + '_grid_pi'
    model = LoadModel(operator_path, shape)
    model.set_init(in_nn)

