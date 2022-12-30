import torch
import numpy as np
from scripts.draw_utils import *

log_list = ['data_based_6_1', 'phys_inc_6_1']
scale_k = 0

x = np.arange(64) / 64 * 2.0
y = np.arange(32) / 32 * 1.0
y, x = np.meshgrid(y, x)
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)
xy_mesh = [x, y, xl, xh, yl, yh]

data_path = 'data/test_data/nse_data_reg_rbc6'
# data_path = 'data/test_data/nse_data_reg_rbc6'
print(data_path)
data = LoadDataRBC2(data_path)
obs, temp, ctr = data.get_data()
# temp = torch.cat((temp, torch.zeros(temp.shape)), dim=-1)
# # temp = torch.cat((torch.zeros(temp.shape), temp), dim=-1)
# print(obs.shape, temp.shape)

# obs_bf = obs[:, :-1]
# obs_af = obs[:, 1:]
# error = rel_error(obs_bf.reshape(-1, 64, 64, 3), obs_af.reshape(-1, 64, 64, 3))
# print(error.mean())
# num = obs.shape[0]
# res = torch.zeros(temp.shape)
# for i in range(num):
#     res[i] = Lpde(obs_bf[i], obs_af[i], 0.01, 0.001, 2.0, 2.0)

# print(res[..., 0].mean(), res[..., 1].mean())
# print(((res[..., 0] - res[..., 0].mean())**2).mean(), ((res[..., 1] - res[..., 1].mean())**2).mean())
# print(res.mean(), temp.mean())
# res = res + temp
# res = res[10*scale_k : 10*(scale_k+1)].mean(0)
# # animate3D(res, xy_mesh, 'rbc', 'obs_e2', zlim=10, dict='rbc')

num_k = 0
animate2D(obs[num_k, ..., 0], xy_mesh, 'u', 'obs', 'rbc')
animate2D(obs[num_k, ..., 1], xy_mesh, 'v', 'obs', 'rbc')
animate2D(obs[num_k, ..., 2], xy_mesh, 'p', 'obs', 'rbc')
# # animate2D(temp[num_k, ..., 1], xy_mesh, 't', 'obs', 'rbc')
animate_field(obs[0, ..., :2], xy_mesh, f'state_6', 'obs', 'rbc')

# anime observation


for file_name in log_list:
    data_list = torch.load(f'logs/data_rbc/output/phase1_test_{file_name}_rbc')
    out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = data_list
    # Lpde_obs = Lpde_obs[10*scale_k : 10*(scale_k+1)].mean(0)
    Lpde_pred = Lpde_pred[10*scale_k : 10*(scale_k+1)].mean(0)
    # Lpde_pred_cul = Lpde_pred_cul[10*scale_k : 10*(scale_k+1)].mean(0)

    # animate_field(out_cul[0, ..., :2], xy_mesh, 'state_6', f'{file_name}', 'rbc')
    # animate2D(out_cul[0, ..., 0], xy_mesh, 'u', file_name, 'rbc')
    # animate2D(out_cul[0, ..., 1], xy_mesh, 'v', file_name, 'rbc')
    # animate2D(out_cul[0, ..., 2], xy_mesh, 'p', file_name, 'rbc')

    # animate3D(Lpde_obs, xy_mesh, 'Lpde_obs', file_name, zlim=1, dict='rbc')
    # animate3D(Lpde_pred, xy_mesh, 'Lpde_pred', file_name, zlim=1, dict='rbc')

#     print(Lpde_obs.shape, temp.shape)
#     Lpde_obs = torch.sqrt((torch.sqrt(Lpde_obs) - temp)**2)
#     Lpde_obs = Lpde_obs[10*scale_k : 10*(scale_k+1)].mean(0)
#     animate3D(Lpde_obs, xy_mesh, 'Lpde_obs', file_name, zlim=5, dict='rbc')


# log_list = ['phys_inc_6', 'data_based_6', 'random_select_0.0001_6', 'no_random_6']
# animate2D_comp(obs, log_list, num_k, xy_mesh, 'comp_rbc', 'rbc', 'rbc')