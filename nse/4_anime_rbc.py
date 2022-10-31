import torch
import numpy as np
from scripts.draw_utils import *

x = np.arange(40) / 40 * 2.0
y = np.arange(40) / 40 * 2.0
x, y = np.meshgrid(x, y)
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)
xy_mesh = [x, y, xl, xh, yl, yh]

data_path = 'data/test_data/nse_data_reg_rbc'
# data_path = 'data/nse_data_reg_rbc'
data = LoadDataRBC(data_path)
obs, temp, ctr = data.get_data()

scale_k = 2
num_k = -1
print(ctr[num_k, 0])
# animate2D(obs[num_k, ..., 0], xy_mesh, 'u', 'obs', 'rbc')
# animate2D(obs[num_k, ..., 1], xy_mesh, 'v', 'obs', 'rbc')
# animate2D(obs[num_k, ..., 2], xy_mesh, 'p', 'obs', 'rbc')
# animate_field(obs[num_k, ..., :2], xy_mesh, 'state', 'obs', 'rbc')

log_list = ['data_based', 'phys_inc', 'no_random', 'random_select_0.001', 'random_select_0.0001', 'prev_phys']
# log_list = ['data_based', 'phys_inc']
for file_name in log_list:
    print(file_name)
    
    data_list = torch.load(f'logs/data_rbc/output/phase1_test_rbc_{file_name}_rbc')
    out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = data_list
    Lpde_obs = Lpde_obs[20*scale_k : 20*(scale_k+1)].mean(0)
    Lpde_pred = Lpde_pred[20*scale_k : 20*(scale_k+1)].mean(0)
    Lpde_pred_cul = Lpde_pred_cul[20*scale_k : 20*(scale_k+1)].mean(0)

    animate_field(out_1step[num_k, ..., :2], xy_mesh, 'state', file_name, 'rbc')
    animate2D(out_cul[num_k, ..., 0], xy_mesh, 'u', file_name, 'rbc')
    animate2D(out_cul[num_k, ..., 1], xy_mesh, 'v', file_name, 'rbc')
    animate2D(out_cul[num_k, ..., 2], xy_mesh, 'p', file_name, 'rbc')

    # animate3D(Lpde_pred, xy_mesh, 'Lpde_pred', file_name, zlim=5, dict = 'rbc')
    # animate3D(Lpde_pred_cul, xy_mesh, 'Lpde_pred_cul', file_name, dict = 'rbc')

    error_1step = (out_1step[num_k] - obs[num_k][1:])[..., :2]
    error_cul = (out_cul[num_k] - obs[num_k][1:])[..., :2]

    # animate3D(error_1step, 'error_1step', file_name, zlim=5)
    animate3D(error_cul, xy_mesh, 'error_cul', file_name, zlim=0.05, dict = 'rbc')



