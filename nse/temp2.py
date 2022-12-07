# from scripts.utils import *

# data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
# data = LoadData(data_path)
# obs, Cd, Cl, ctr = data.split(1, 5)

# obs_bf = obs[:, :-1]
# obs_af = obs[:, 1:]

# file_name = 'data_based'
# out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = torch.load(f'logs/data/output/phase1_test_{file_name}_fb_0.0')

# print(rel_error(obs_bf, obs_af).mean())
# print(rel_error(out_1step, obs_af).mean())
# print(rel_error(out_cul, obs_af).mean())

# file_name = 'baseline'
# out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = torch.load(f'logs/data/output/phase1_test_{file_name}_fb_0.0')

# print(rel_error(obs_bf, obs_af).mean())
# print(rel_error(out_1step, obs_af).mean())
# print(rel_error(out_cul, obs_af).mean())

from scripts.utils import *

data_path = 'data/test_data/nse_data_reg_rbc'
data = LoadDataRBC(data_path)
obs, temp, ctr = data.get_data()

obs_bf = obs[:, :-1]
obs_af = obs[:, 1:]

file_name = 'data_based'
out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = torch.load(f'logs/data_rbc/output/phase1_test_rbc_{file_name}_rbc')

print(rel_error(obs_bf, obs_af).mean())
print(rel_error(out_1step, obs_af).mean())
print(rel_error(out_cul, obs_af).mean())

file_name = 'phys_inc'
out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = torch.load(f'logs/data_rbc/output/phase1_test_rbc_{file_name}_rbc')

print(rel_error(obs_bf, obs_af).mean())
print(rel_error(out_1step, obs_af).mean())
print(rel_error(out_cul, obs_af).mean())


import torch
import numpy as np
from scripts.draw_utils import *

data_path = 'data/nse_data_reg_rbc2_test'
data = LoadDataRBC1(data_path)
obs, temp, ctr = data.get_data()
temp = torch.cat((temp, torch.zeros(temp.shape)), dim=-1)
# temp = torch.cat((torch.zeros(temp.shape), temp), dim=-1)
print(obs.shape, temp.shape)

obs_bf = obs[:, :-1]
obs_af = obs[:, 1:]
error = rel_error(obs_bf.reshape(-1, 64, 64, 3), obs_af.reshape(-1, 64, 64, 3))
print(error)


import torch
import numpy as np
from scripts.draw_utils import *

data_path = 'data/nse_data_reg_rbc_rest'
data = LoadDataRBC(data_path)
obs, temp, ctr = data.get_data()
temp = torch.cat((temp, torch.zeros(temp.shape)), dim=-1)
# temp = torch.cat((torch.zeros(temp.shape), temp), dim=-1)
print(obs.shape, temp.shape)

obs_bf = obs[:, :-1]
obs_af = obs[:, 1:]
error = rel_error(obs_bf.reshape(-1, 64, 64, 3), obs_af.reshape(-1, 64, 64, 3))
print(error)

x = np.arange(64) / 64 * 2.0
y = np.arange(64) / 64 * 2.0
x, y = np.meshgrid(x, y)
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)
xy_mesh = [x, y, xl, xh, yl, yh]

animate_field(obs[0, ..., :2], xy_mesh, 'state_4', 'obs', 'rbc')