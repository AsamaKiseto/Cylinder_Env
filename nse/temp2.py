from scripts.utils import *

data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadData(data_path)
obs, Cd, Cl, ctr = data.split(1, 5)

obs_bf = obs[:, :-1]
obs_af = obs[:, 1:]

file_name = 'data_based'
out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = torch.load(f'logs/data/output/phase1_test_{file_name}_fb_0.0')

print(rel_error(obs_bf, obs_af).mean())
print(rel_error(out_1step, obs_af).mean())
print(rel_error(out_cul, obs_af).mean())

file_name = 'baseline'
out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul = torch.load(f'logs/data/output/phase1_test_{file_name}_fb_0.0')

print(rel_error(obs_bf, obs_af).mean())
print(rel_error(out_1step, obs_af).mean())
print(rel_error(out_cul, obs_af).mean())