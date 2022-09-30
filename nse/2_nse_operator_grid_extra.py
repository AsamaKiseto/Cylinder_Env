import torch

from scripts.utils import *
from scripts.nse_model import *

if __name__=='__main__':
    # logs
    logs_state = 'logs/phase1_ex1_grid_pi'
    logs_fname = 'logs/phase1_test_grid_pi'
    pred_log, phys_log, model_log = torch.load(logs_state)
    args = model_log['args']
    args.epochs = 100
    args.step_size = 20
    args.lr = 1e-4

    logs = dict()
    logs['args'] = args

    # load data
    data_path = 'data/' + args.data_path + '_extra'
    tg = args.tg     # sample evrey 5 timestamps
    Ng = args.Ng
    data = LoadData(data_path)
    obs, Cd, Cl, ctr = data.split(Ng, tg)
    logs['data_norm'] = data.normalize(method = 'logs_norm', logs = model_log)
    logs['pred_model'] = []
    logs['phys_model'] = []

    logs['test_loss_trans']=[]
    logs['test_loss_u_t_rec']=[]
    logs['test_loss_ctr_t_rec']=[]
    logs['test_loss_trans_latent']=[]
    logs['test_loss_pde_obs'] = []
    logs['test_loss_pde_pred'] = []

    # data param
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]

    # loader
    train_loader, test_loader = data.trans2TrainingSet(args.batch_size)

    # model setting
    nse_model = NSEModel_FNO(args, shape, data.dt)
    params_num = nse_model.count_params()

    nse_model.load_state(pred_log, phys_log)

    print('N0: {}, nt: {}, nx: {}, ny: {}, device: {}'.format(N0, nt, nx, ny, nse_model.device))
    print(f'Cd: {logs["data_norm"]["Cd"]}')
    print(f'Cl: {logs["data_norm"]["Cl"]}')
    print(f'ctr: {logs["data_norm"]["ctr"]}')
    print(f'obs: {logs["data_norm"]["obs"]}')
    print(f'param numbers of the model: {params_num}')

    nse_model.process_extra(train_loader, test_loader, logs)
    torch.save([nse_model.pred_model.state_dict(), nse_model.phys_model.state_dict(), logs], logs_fname)