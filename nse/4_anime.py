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
obs, _, _, ctr = torch.load('data/test_data/nse_data_reg_dt_0.01_fb_0.0')
obs = obs[:, ::tg][:, 1:][..., 2:]
ctr = ctr[:, ::tg]

nt, nx, ny = obs.shape[1], obs.shape[2], obs.shape[3]
scale_k = 2
num_k = -1

# log_list = ['data_based', 'phys_inc', 'no_random', 'random_select_0.001']
# for file_name in log_list:
#     print(file_name)
    
#     data_list = torch.load(f'logs/data_nse/output/phase1_test_{file_name}_fb_0.0')
#     out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = data_list
#     Lpde_obs = Lpde_obs[10*scale_k : 10*(scale_k+1)].mean(0)
#     Lpde_pred = Lpde_pred[10*scale_k : 10*(scale_k+1)].mean(0)
#     # Lpde_pred_cul = Lpde_pred_cul[10*scale_k : 10*(scale_k+1)].mean(0)

#     # animate2D(out_cul[0, ..., 0], xy_mesh, 'u', file_name)
#     # animate2D(out_cul[0, ..., 1], xy_mesh, 'v', file_name)
#     # animate2D(out_cul[0, ..., 2], xy_mesh, 'p', file_name)

#     # animate3D(Lpde_pred, xy_mesh, 'Lpde_pred', file_name, zlim=5)
#     # animate3D(Lpde_pred_cul, xy_mesh, 'Lpde_pred_cul', file_name)

#     k = 0
#     error_cul = (out_cul[k] - obs[k])[..., :2]

#     # animate3D(error_cul, xy_mesh, 'error_cul', file_name, zlim=5)

log_list = ['phys_inc', 'data_based']
# animate2D_comp(obs, log_list, -1, xy_mesh, 'comp1')

log_list = ['phys_inc', 'random_select_0.001', 'no_random']
animate2D_comp(obs, log_list, -1, xy_mesh, 'comp2')