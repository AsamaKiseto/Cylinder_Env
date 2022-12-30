import torch
import numpy as np
from scripts.draw_utils import *

x = np.arange(256) / 256 * 2.2
y = np.arange(64) / 64 * 0.41
y, x = np.meshgrid(y, x)
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)
xy_mesh = [x, y, xl, xh, yl, yh]

tg = 5
scale_k = 2
num_k = -1

# obs, _, _, ctr = torch.load('data/test_data/nse_data_reg_dt_0.01_fb_0.0')
# obs = obs[:, ::tg][..., 2:]
# obs_af = obs[num_k, 1:]
# obs_bf = obs[num_k, :-1]
# obs = obs[:, 1:]
# ctr = ctr[:, ::tg]

# nt, nx, ny = obs.shape[1], obs.shape[2], obs.shape[3]

# Loss_pde = Lpde(obs_bf, obs_af, 0.05)
# loss = torch.sqrt(Loss_pde[..., 0]**2 + Loss_pde[..., 1]**2)
# animate3D(Loss_pde, xy_mesh, 'obs_Loss', 'pde', zlim=100)

# animate_field(obs[num_k], xy_mesh, 'obs', 'state')

log_list = ['data_based', 'phys_inc']
for file_name in log_list:
    print(file_name)
    
    data_list = torch.load(f'logs/data_nse/output/phase1_test_{file_name}_fb_0.0')
    out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = data_list
    Lpde_obs = Lpde_obs[10*scale_k : 10*(scale_k+1)].mean(0)
    Lpde_pred = Lpde_pred[10*scale_k : 10*(scale_k+1)].mean(0)
    Lpde_pred_cul = Lpde_pred_cul[10*scale_k : 10*(scale_k+1)].mean(0)

    # animate2D(out_cul[0, ..., 0], xy_mesh, 'u', file_name)
    # animate2D(out_cul[0, ..., 1], xy_mesh, 'v', file_name)
    # animate2D(out_cul[0, ..., 2], xy_mesh, 'p', file_name)

    animate3D(Lpde_pred, xy_mesh, 'Lpde_pred', file_name, zlim=5)
    animate3D(Lpde_pred_cul, xy_mesh, 'Lpde_pred_cul', file_name)

#     k = 0
#     error_cul = (out_cul[k] - obs[k])[..., :2]

#     # animate3D(error_cul, xy_mesh, 'error_cul', file_name, zlim=5)

# log_list = ['phys_inc', 'data_based']
# animate2D_comp(obs, log_list, num_k, xy_mesh, 'comp1')

log_list = ['phys_inc', 'data_based', 'random_select_0.001', 'no_random']
animate2D_comp(obs, log_list, num_k, xy_mesh, 'comp2')