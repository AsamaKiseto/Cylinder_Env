import numpy as np
import matplotlib.pyplot as plt 
import torch

from scripts.models import *
from scripts.utils import *
from scripts.draw_utils import *


# data param
nx, ny = 256, 64
shape = [nx, ny]
dt = 0.01

logs = torch.load('logs/phase1_env_logs_1')
scale, obs, Cd, Cl = logs['scale'], np.asarray(logs['obs']), np.asarray(logs['Cd']), np.asarray(logs['Cl'])

k = 1
tg = 5
t_start = 5
nt = 80
t_nn = (np.arange(nt) + 1) * 0.01 * tg
label = [1, 2]
obs_sps, Cd_sps, Cl_sps = obs[:, ::tg][...,2:], Cd[:, tg-1::tg], Cl[:, tg-1::tg]

obs_sps1, obs_sps2 = obs_sps[::2][:, 1:], obs_sps[1::2][:, 1:]
Cd_sps1, Cd_sps2 = Cd_sps[::2], Cd_sps[1::2]
Cl_sps1, Cl_sps2 = Cl_sps[::2], Cl_sps[1::2]

ex_name = 'ex0'
obs_nn, Cd_nn, Cl_nn = logs[ex_name]['obs_nn'], logs[ex_name]['Cd_nn'], logs[ex_name]['Cl_nn']
obs_nn, Cd_nn, Cl_nn = np.asarray(obs_nn), np.asarray(Cd_nn), np.asarray(Cl_nn)

Lpde_obs, Lpde_nn = logs[ex_name]['Lpde_obs'], logs[ex_name]['Lpde_nn']
Lpde_obs, Lpde_nn = np.asarray(Lpde_obs), np.asarray(Lpde_nn)

obs_nn1, obs_nn2 = obs_nn[::2], obs_nn[1::2]
Cd_nn1, Cd_nn2 = Cd_nn[::2], Cd_nn[1::2]
Cl_nn1, Cl_nn2 = Cl_nn[::2], Cl_nn[1::2]
Lpde_obs1, Lpde_obs2 = Lpde_obs[::2], Lpde_obs[1::2]
Lpde_nn1, Lpde_nn2 = Lpde_nn[::2], Lpde_nn[1::2]

size = obs_sps1.shape[0]


error1_1 = ((obs_nn1 - obs_sps1) ** 2).reshape(size, nt, -1).mean(2).mean(0)
error1_2 = ((Cd_nn1 - Cd_sps1) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error1_3 = ((Cl_nn1 - Cl_sps1) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error1_4 = Lpde_obs1.reshape(size, nt, -1).mean(2).mean(0)
error1_5 = Lpde_nn1.reshape(size, nt, -1).mean(2).mean(0)
error2_1 = ((obs_nn2 - obs_sps2) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error2_2 = ((Cd_nn2 - Cd_sps2) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error2_3 = ((Cl_nn2 - Cl_sps2) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error2_4 = Lpde_obs2.reshape(size, nt, -1).mean(2).mean(0)
error2_5 = Lpde_nn2.reshape(size, nt, -1).mean(2).mean(0)

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

for i in label:
    for j in range(5):
        exec(f'ax[j].plot(t_nn[t_start:], error{i}_{j+1}[t_start:], label="data-based")')
        ax[j].legend()


ex_name = 'ex1_3'
obs_nn, Cd_nn, Cl_nn = logs[ex_name]['obs_nn'], logs[ex_name]['Cd_nn'], logs[ex_name]['Cl_nn']
obs_nn, Cd_nn, Cl_nn = np.asarray(obs_nn), np.asarray(Cd_nn), np.asarray(Cl_nn)

Lpde_obs, Lpde_nn = logs[ex_name]['Lpde_obs'], logs[ex_name]['Lpde_nn']
Lpde_obs, Lpde_nn = np.asarray(Lpde_obs), np.asarray(Lpde_nn)

obs_nn0, obs_nn1, obs_nn2, obs_nn3 = obs_nn[0], obs_nn[1::3], obs_nn[2::3], obs_nn[3::3]
Cd_nn0, Cd_nn1, Cd_nn2, Cd_nn3 = Cd_nn[0], Cd_nn[1::3], Cd_nn[2::3], Cd_nn[3::3]
Cl_nn0, Cl_nn1, Cl_nn2, Cl_nn3 = Cl_nn[0], Cl_nn[1::3], Cl_nn[2::3], Cl_nn[3::3]
Lpde_obs0, Lpde_obs1, Lpde_obs2, Lpde_obs3 = Lpde_obs[0], Lpde_obs[1::3], Lpde_obs[2::3], Lpde_obs[3::3]
Lpde_nn0, Lpde_nn1, Lpde_nn2, Lpde_nn3 = Lpde_nn[0], Lpde_nn[1::3], Lpde_nn[2::3], Lpde_nn[3::3]

size = obs_sps1.shape[0]

error0_1 = ((obs_nn0 - obs_sps0) ** 2).reshape(nt, -1).mean(1) 
error0_2 = ((Cd_nn0 - Cd_sps0) ** 2).reshape(nt, -1).mean(1) 
error0_3 = ((Cl_nn0 - Cl_sps0) ** 2).reshape(nt, -1).mean(1) 
error0_4 = Lpde_obs0.reshape(nt, -1).mean(1) 
error0_5 = Lpde_nn0.reshape(nt, -1).mean(1) 
error1_1 = ((obs_nn1 - obs_sps1) ** 2).reshape(size, nt, -1).mean(2).mean(0)
error1_2 = ((Cd_nn1 - Cd_sps1) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error1_3 = ((Cl_nn1 - Cl_sps1) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error1_4 = Lpde_obs1.reshape(size, nt, -1).mean(2).mean(0)
error1_5 = Lpde_nn1.reshape(size, nt, -1).mean(2).mean(0)
error2_1 = ((obs_nn2 - obs_sps2) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error2_2 = ((Cd_nn2 - Cd_sps2) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error2_3 = ((Cl_nn2 - Cl_sps2) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error2_4 = Lpde_obs2.reshape(size, nt, -1).mean(2).mean(0)
error2_5 = Lpde_nn2.reshape(size, nt, -1).mean(2).mean(0)
error3_1 = ((obs_nn3 - obs_sps3) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error3_2 = ((Cd_nn3 - Cd_sps3) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error3_3 = ((Cl_nn3 - Cl_sps3) ** 2).reshape(size, nt, -1).mean(2).mean(0) 
error3_4 = Lpde_obs3.reshape(size, nt, -1).mean(2).mean(0)
error3_5 = Lpde_nn3.reshape(size, nt, -1).mean(2).mean(0)

# fig setting

for i in label:
    for j in range(5):
        exec(f'ax[j].plot(t_nn[t_start:], error{i}_{j+1}[t_start:], label="phys_included")')
        ax[j].legend()


plt.savefig(f'logs/coef_phase1_test_{label[0]}.jpg')


from scripts.utils import *
data_path = 'data/nse_data_reg'
data = LoadData(data_path)
data.normalize('unif')
