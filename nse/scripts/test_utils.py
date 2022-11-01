import numpy as np
import torch
from scripts.models import *

def loss_log(data, file_name, test_rate = 0.2):
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]
    _, _, model_log = torch.load('logs/phase1_' + file_name)
    args, pred_model, phys_model = model_log['args'], model_log['pred_model'], model_log['phys_model']
    epochs = len(pred_model)

    data.normalize('logs_unif', model_log)
    data_loader = data.trans2CheckSet(test_rate, args.batch_size)
    model = NSEModel_FNO(shape, data.dt, args)
    loss = np.zeros((3, epochs))

    loss_log = dict()
    loss_log['test_loss_trans']=[]
    loss_log['test_loss_u_t_rec']=[]
    loss_log['test_loss_ctr_t_rec']=[]
    loss_log['test_loss_trans_latent']=[]
    loss_log['test_loss_pde_obs'] = []
    loss_log['test_loss_pde_pred'] = []

    print('begin simulation')
    for i in range(epochs):
        t1 = default_timer()
        model.load_state(pred_model[i], phys_model[i])
        model.test(data_loader, loss_log)
        t2 = default_timer()
        print(f'# {i+1} : {t2 - t1} | {loss_log["test_loss_trans"][-1]} | {loss_log["test_loss_pde_obs"][-1]} | {loss_log["test_loss_pde_pred"][-1]}')
    loss[0] = np.asarray(loss_log['test_loss_trans'])
    loss[1] = np.asarray(loss_log['test_loss_pde_obs'])
    loss[2] = np.asarray(loss_log['test_loss_pde_pred'])
    print('end simulation')

    data.unnormalize()
    torch.save(loss, 'logs/data/losslog/loss_log_' + file_name)

def test_log(data, file_name, ex_name, model_loaded = NSEModel_FNO, dict = 'nse', dt = 0.05):
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]

    operator_path = f'logs/model_{dict}/phase1_{file_name}'
    
    print(operator_path)
    state_dict_pred, state_dict_phys, logs = torch.load(operator_path)
    data.normalize('logs_unif', logs)

    model = model_loaded(shape, dt, logs['args'])
    model.load_state(state_dict_pred, state_dict_phys)
    model.toCPU()
    
    obs = data.get_obs()

    error_1step, Lpde_obs, Lpde_pred = model.cal_1step(data)
    out_cul, Lpde_pred_cul = model.process(data)
    
    error_cul = ((out_cul - obs[:, 1:]) ** 2).reshape(N0, nt, -1).mean(2)
    # print(f'Lpde_nn: {Lpde_pred_cul[-1]}')
    
    data.unnormalize()
    log_data = [out_cul, Lpde_obs, Lpde_pred, Lpde_pred_cul]
    log_error = [error_1step, error_cul]
    
    torch.save(log_data, f'logs/data_{dict}/output/phase1_test_{file_name}_{ex_name}')
    torch.save(log_error, f'logs/data_{dict}/error/phase1_test_{file_name}_{ex_name}')
