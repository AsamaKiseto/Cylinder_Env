import numpy as np
import matplotlib.pyplot as plt 
import torch

from scripts.nse_model import *
from scripts.utils import *

dt = 0.01
tg = 5

# ex_nums = ['baseline', 'ps_0.01', 'ps_0.03', 'ps_0.08', 'pe_20', 'pe_30', 'pe_40', 'pe_50']
# ex_nums = ['data_based', 'ps_0.1']
ex_nums = ['data_based', 'ps_0.1']
# ex_nums = ['ps_0.01', 'ps_0.03', 'baseline', 'ps_0.08', 'ps_0.1']
scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_model = len(ex_nums)
        
if __name__ == '__main__':
    # load test data
    test_data_name = '_fb_0.0'
    data_path = 'data/test_data/nse_data_reg_dt_0.01' + test_data_name

    # data_path = 'data/nse_data_reg_dt_0.01_fr_1.0'
    print('load data')
    data = LoadData(data_path)
    data.split(1, tg)
    N0, nt, nx, ny = data.get_params()
    print('load data finished')

    print(N0, nt, nx, ny)
    shape = [nx, ny]
    t_nn = (np.arange(nt) + 1) * 0.01 * tg
    t = (np.arange(nt * tg) + 1) * 0.01 
    
    ts = 2.0 // (dt * tg)

    operator_path = 'logs/phase1_' + ex_nums[0] + '_grid_pi'
    model = LoadModel(operator_path, shape)
    
    data.normalize('logs_unif', model.data_norm)
    print(model.data_norm)
    obs, Cd, Cl, ctr = data.get_data()
    in_nn = obs[:, 0]
    model.set_init(in_nn)
    
    out_init, Lpde_pred_init, error_Cd_init, error_Cl_init = model.process(obs[:, :ts + 1], Cd[:, :ts], Cl[:, :ts], ctr[:, :ts])
    print(f'out_init.shape: {out_init.shape}')
    in_nn = out_init[:, -1]
    
    data.unnormalize()
    
    for k in range(n_model):
        operator_path = 'logs/phase1_' + ex_nums[k] + '_grid_pi'
        model = LoadModel(operator_path, shape)
        data.normalize('logs_unif', model.data_norm)
        print(model.data_norm)
        obs, Cd, Cl, ctr = data.get_data()
        
        # in_nn = obs[:, 0]
        model.set_init(in_nn)

        out_cul, Lpde_pred_cul, error_Cd_cul, error_Cl_cul = model.process(obs[:, ts:], Cd[:, ts:], Cl[:, ts:], ctr[:, ts:])
        print(f'out_cul.shape: {out_cul.shape}')
        # print(f'Lpde_nn: {Lpde_pred_cul[-1]}')
        
        # print(f'Lpde_obs: {Lpde_obs[-1]}')
        # print(f'Lpde_pred: {Lpde_pred[-1]}')

        data.unnormalize()
        out_cul = torch.cat((out_init, out_cul), dim=1)
        Lpde_pred_cul = torch.cat((Lpde_pred_init, Lpde_pred_cul), dim=1)
        error_Cd_cul = torch.cat((error_Cd_init, error_Cd_cul), dim=1)
        error_Cl_cul = torch.cat((error_Cl_init, error_Cl_cul), dim=1)
        
        log_data = [out_cul, Lpde_pred_cul, error_Cd_cul, error_Cl_cul]
        torch.save(log_data, 'logs/data/phase1_test_ts_2.0_' + ex_nums[k] + test_data_name)
