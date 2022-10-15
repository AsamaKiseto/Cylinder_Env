import numpy as np
import torch
from scripts.nse_model import *

def loss_log(data, file_name, test_rate = 0.1):
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]
    _, _, model_log = torch.load('logs/phase1_' + file_name + '_grid_pi')
    args, data_norm = model_log['args'], model_log['data_norm']
    pred_model, phys_model = model_log['pred_model'], model_log['phys_model']
    epochs = len(pred_model)

    data.normalize('logs_unif', data_norm)
    data_loader = data.trans2CheckSet(test_rate, args.batch_size)
    model = NSEModel_FNO(shape, data.dt, args)
    loss = np.zeros((3, epochs))

    print('begin simulation')
    for i in range(epochs):
        t1 = default_timer()
        model.load_state(pred_model[i], phys_model[i])
        loss[0, i], _, _, _, loss[1, i], loss[2, i] = model.simulate(data_loader)
        t2 = default_timer()
        print(f'# {i+1} : {t2 - t1} | {loss[0, i].mean()} | {loss[1, i].mean()} | {loss[2, i].mean()}')
    print('end simulation')

    data.unnormalize()
    torch.save(loss, 'logs/data/losslog/loss_log_' + file_name)

def test_log(data, file_name, ex_name):
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]
    operator_path = 'logs/phase1_' + file_name + '_grid_pi'
    model = LoadModel(operator_path, shape)
    data.normalize('logs_unif', model.data_norm)
    obs, Cd, Cl, ctr = data.get_data()
    in_nn = obs[:, 0]
    
    model.set_init(in_nn)

    out_1step, Lpde_obs, Lpde_pred, error_Cd_1step, error_Cl_1step = model.cal_1step(obs, Cd, Cl, ctr)
    out_cul, Lpde_pred_cul, error_Cd_cul, error_Cl_cul = model.process(obs, Cd, Cl, ctr)
    
    error_1step = ((out_1step - obs[:, 1:]) ** 2).reshape(N0, nt, -1).mean(2)
    error_cul = ((out_cul - obs[:, 1:]) ** 2).reshape(N0, nt, -1).mean(2)
    # print(f'Lpde_nn: {Lpde_pred_cul[-1]}')
    
    data.unnormalize()
    log_data = [out_1step, out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul]
    log_error = [error_1step, error_cul, error_Cd_1step, error_Cl_1step, error_Cd_cul, error_Cl_cul]
    
    torch.save(log_data, f'logs/data/output/phase1_test_{file_name}_{ex_name}')
    torch.save(log_error, f'logs/data/error/phase1_test_{file_name}_{ex_name}')
