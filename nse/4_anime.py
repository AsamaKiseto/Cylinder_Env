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
obs, _, _, ctr = torch.load('data/test_data/nse_data_reg_dt_0.01_fb_0.0_scale_0.1')
obs = obs[:, ::tg][:, 1:][..., 2:]
ctr = ctr[:, ::tg]

# animate2D(obs[0, ..., 0], xy_mesh, 'u', 'obs')
# animate2D(obs[0, ..., 1], xy_mesh, 'v', 'obs')
# animate2D(obs[0, ..., 2], xy_mesh, 'p', 'obs')
animate_field(obs[0, ..., :2], xy_mesh, 'state', 'obs')

log_list = ['data_based', 'baseline', 'no_random', 'random_select_0.01', 'random_select_0.001', 'pre_phys']
for file_name in log_list:
    print(file_name)
    
    data_list = torch.load(f'logs/data_bak/output/phase1_test_{file_name}_fb_0.0')
    out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = data_list
    Lpde_obs = Lpde_obs[:10].mean(0)
    Lpde_pred = Lpde_pred[:10].mean(0)
    Lpde_pred_cul = Lpde_pred_cul[:10].mean(0)

    animate_field(out_1step[0, ..., :2], xy_mesh, 'state', file_name)
    # animate2D(out_cul[0, ..., 0], xy_mesh, 'u', file_name)
    # animate2D(out_cul[0, ..., 1], xy_mesh, 'v', file_name)
    # animate2D(out_cul[0, ..., 2], xy_mesh, 'p', file_name)

    # animate3D(Lpde_pred, xy_mesh, 'Lpde_pred', file_name, zlim=5)
    # animate3D(Lpde_pred_cul, xy_mesh, 'Lpde_pred_cul', file_name)

    k = 0
    error_1step = (out_1step[k] - obs[k])[..., :2]
    error_cul = (out_cul[k] - obs[k])[..., :2]

    # animate3D(error_1step, xy_mesh, 'error_1step', file_name, zlim=5)
    animate3D(error_cul, xy_mesh, 'error_cul', file_name, zlim=5)

# ts_list = ['0.5', '1.0', '1.5']
# for ts in ts_list:
#     file_name = 'baseline'
#     print(file_name)
    
#     data_list = torch.load(f'logs/data/phase1_test_ts_{ts}_baseline_fb_0.0')
#     out_cul, Lpde_pred_cul, _, _ = data_list
#     Lpde_pred_cul = Lpde_pred_cul[:10].mean(0)

#     animate(out_cul[0, ..., :2], 'out_cul_test', file_name)

#     # animate3D(Lpde_pred_cul, 'Lpde_pred_cul_test', file_name)

#     k = 0
#     error_cul = (out_cul[k] - obs[k])[..., :2]

#     # animate3D(error_cul, f'error_cul_test_ts_{ts}', file_name, zlim=5)

# file_name = 'baseline'
# print(file_name)

# data_list = torch.load(f'logs/data/phase1_test_data_based_fb_0.0')
# out_cul, Lpde_pred_cul, _, _ = data_list
# Lpde_pred_cul = Lpde_pred_cul[:10].mean(0)

# animate(out_1step[0, ..., :2], 'out_1step', file_name)
# animate(out_cul[0, ..., :2], 'out_cul', file_name)

# animate3D(Lpde_pred_cul, 'Lpde_pred_cul_test', file_name)

# k = 0
# error_cul = (out_cul[k] - obs[k])[..., :2]

# animate3D(error_cul, 'error_cul_test', file_name, zlim=5)


