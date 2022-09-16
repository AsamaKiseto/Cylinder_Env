import torch
import argparse

from scripts.utils import *
from scripts.nse_model import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('-dp', '--data_path', default='nse_data_reg', type=str, help='data path name')
    parser.add_argument('-lf', '--logs_fname', default='test', type=str, help='logs file name')
    
    parser.add_argument('-L', '--L', default=2, type=int, help='the number of layers')
    parser.add_argument('-m', '--modes', default=16, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('-w', '--width', default=32, type=int, help='the number of width of FNO layer')
    

    parser.add_argument('--phys_gap', default=10, type=int, help = 'Number of gap of Phys')
    parser.add_argument('--phys_epochs', default=1, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_steps', default=3, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_scale', default=0.01, type=float, help = 'Number of Phys Epochs')

    parser.add_argument('-bs', '--batch_size', default=64, type=int, help = 'batch size')
    parser.add_argument('-eps', '--epochs', default=500, type=int, help = 'Number of Epochs')
    parser.add_argument('-lr', '--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--gpu', default=0, type=int, help='device number')

    parser.add_argument('-tg', '--tg', default=5, type=int, help = 'time gap')
    parser.add_argument('-Ng', '--Ng', default=1, type=int, help = 'N gap')
    parser.add_argument('-l1', '--lambda1', default=1, type=float, help='weight of losses1')
    parser.add_argument('-l2', '--lambda2', default=0.1, type=float, help='weight of losses2')
    parser.add_argument('-l3', '--lambda3', default=0.1, type=float, help='weight of losses3')
    parser.add_argument('-l4', '--lambda4', default=0.1, type=float, help='weight of losses4')
    parser.add_argument('-l5', '--lambda5', default=1, type=float, help='weight of losses5')
    parser.add_argument('-fc', '--f_channels', default=1, type=int, help='channels of f encode')
    
    return parser.parse_args(argv)

if __name__=='__main__':
    # args parser
    args = get_args()
    print(args)

    # logs
    logs = dict()
    logs['args'] = args

    logs_fname = 'logs/phase1_' + args.logs_fname + '_grid_pi'

    # load data
    data_path = 'data/' + args.data_path
    tg = args.tg     # sample evrey 20 timestamps
    Ng = args.Ng
    data = ReadData(data_path)
    obs, Cd, Cl, ctr = data.split(Ng, tg)
    obs_bf = obs[:, :-1]
    obs_af = obs[:, 1:]
    logs['data_norm'] = data.norm()
    logs['logs'] = dict()
    logs['logs']['train_loss']=[]
    logs['logs']['train_loss_trans']=[]
    logs['logs']['train_loss_u_t_rec']=[]
    logs['logs']['train_loss_f_t_rec']=[]
    logs['logs']['train_loss_trans_latent']=[]
    logs['logs']['train_loss_pde'] = []
    logs['logs']['test_loss']=[]
    logs['logs']['test_loss_trans']=[]
    logs['logs']['test_loss_u_t_rec']=[]
    logs['logs']['test_loss_f_t_rec']=[]
    logs['logs']['test_loss_trans_latent']=[]
    logs['logs']['test_loss_pde'] = []

    # data param
    N0, nt, nx, ny = data.get_params()
    shape = [nx, ny]

    # loader
    train_loader, test_loader = data.trans2Dataset(args.batch_size)

    # model setting
    nse_model = NSEModel_FNO(args, shape, data.dt, logs['logs'])
    params_num = nse_model.count_params()

    print('N0: {}, nt: {}, nx: {}, ny: {}, device: {}'.format(N0, nt, nx, ny, nse_model.device))
    print(f'Cd: {logs["data_norm"]["Cd"]}')
    print(f'Cl: {logs["data_norm"]["Cl"]}')
    print(f'ctr: {logs["data_norm"]["ctr"]}')
    print(f'obs: {logs["data_norm"]["obs"]}')
    print(f'param numbers of the model: {params_num}')

    nse_model.process(train_loader, test_loader)
    torch.save([nse_model.model.state_dict(), logs], logs_fname)