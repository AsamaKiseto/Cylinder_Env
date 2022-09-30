import torch
import argparse

from scripts.utils import *
from scripts.nse_model import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('-dp', '--data_path', default='nse_data_reg_extra', type=str, help='data path name')
    parser.add_argument('-lf', '--logs_fname', default='test', type=str, help='logs file name')
    
    parser.add_argument('--phys_gap', default=20, type=int, help = 'Number of gap of Phys')
    parser.add_argument('--phys_epochs', default=10, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_steps', default=2, type=int, help = 'Number of Phys Steps')
    parser.add_argument('--phys_scale', default=0.05, type=float, help = 'Number of Phys Scale')

    parser.add_argument('--batch_size', default=64, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=100, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--step_size', default=20, type=int, help='scheduler step size')
    
    return parser.parse_args(argv)

if __name__=='__main__':
    # args parser
    ipt_args = get_args()
    print(ipt_args)

    # new args params
    logs_state = 'logs/phase1_' + ipt_args.logs_fname + '_grid_pi'
    logs_fname = 'logs/phase1_' + ipt_args.logs_fname + '_extra_grid_pi'
    pred_log, phys_log, model_log = torch.load(logs_state)
    args = model_log['args']
    args.data_path = ipt_args.data_path
    args.logs_fname = ipt_args.logs_fname + '_extra'

    args.phys_gap = ipt_args.phys_gap
    args.phys_epochs = ipt_args.phys_epochs
    args.phys_scale = ipt_args.phys_scale

    args.batch_size = ipt_args.batch_size
    args.epochs = ipt_args.epochs
    args.step_size = ipt_args.step_size
    args.lr = ipt_args.lr

    args.phys_scale = ipt_args.phys_scale
    
    # log setting
    logs = dict()
    logs['args'] = args

    # load data
    data_path = 'data/' + args.data_path
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
    nse_model = NSEModel_FNO(shape, data.dt, args)
    params_num = nse_model.count_params()

    nse_model.load_state(pred_log, phys_log)

    print('N0: {}, nt: {}, nx: {}, ny: {}, device: {}'.format(N0, nt, nx, ny, nse_model.device))
    print(f'Cd: {logs["data_norm"]["Cd"]}')
    print(f'Cl: {logs["data_norm"]["Cl"]}')
    print(f'ctr: {logs["data_norm"]["ctr"]}')
    print(f'obs: {logs["data_norm"]["obs"]}')
    print(f'param numbers of the model: {params_num}')

    # extra train process
    for epoch in range(1, nse_model.params.epochs+1):
        nse_model.data_train(epoch, train_loader)
        if epoch % nse_model.params.phys_gap == 0 and epoch != nse_model.params.epochs:
            # freeze phys_model trained in data training
            for param in list(nse_model.phys_model.parameters()):
                param.requires_grad = False

            for phys_epoch in range(1, nse_model.params.phys_epochs+1):
                nse_model.phys_train(phys_epoch, train_loader)
            
            for param in list(nse_model.phys_model.parameters()):
                param.requires_grad = True
        nse_model.save_log(logs)
        nse_model.test(test_loader, logs)

    torch.save([nse_model.pred_model.state_dict(), nse_model.phys_model.state_dict(), logs], logs_fname)