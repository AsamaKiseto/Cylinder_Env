import torch
import argparse

from scripts.utils import *
from scripts.nse_model import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('-lf', '--logs_fname', default='test', type=str, help='logs file name')
    parser.add_argument('-dr', '--data_rate', default=0.7, type=float, help='logs file name')
    parser.add_argument('-dc', '--dict', default='model_rbc', type=str, help='dict name')
    
    parser.add_argument('-L', '--L', default=2, type=int, help='the number of layers')
    parser.add_argument('-m', '--modes', default=16, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('-w', '--width', default=32, type=int, help='the number of width of FNO layer')
    
    parser.add_argument('--phys_gap', default=2, type=int, help = 'Number of gap of Phys')
    parser.add_argument('--phys_epochs', default=2, type=int, help = 'Number of Phys Epochs')
    parser.add_argument('--phys_steps', default=2, type=int, help = 'Number of Phys Steps')
    parser.add_argument('--phys_scale', default=0.1, type=float, help = 'Number of Phys Scale')
    parser.add_argument('--phys_random_select', default=False, type=bool, help = 'Whether random select')

    parser.add_argument('--batch_size', default=32, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=500, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--gpu', default=0, type=int, help='device number')

    parser.add_argument('-tg', '--tg', default=1, type=int, help = 'time gap')
    parser.add_argument('-Ng', '--Ng', default=2, type=int, help = 'N gap')
    parser.add_argument('-l1', '--lambda1', default=1, type=float, help='weight of losses1')
    parser.add_argument('-l2', '--lambda2', default=0.1, type=float, help='weight of losses2')
    parser.add_argument('-l3', '--lambda3', default=0.05, type=float, help='weight of losses3')
    parser.add_argument('-l4', '--lambda4', default=0.5, type=float, help='weight of losses4')
    parser.add_argument('-fc', '--f_channels', default=1, type=int, help='channels of f encode')
    
    return parser.parse_args(argv)

if __name__=='__main__':
    # args parser
    args = get_args()
    print(args)

    # logs
    logs = dict()
    logs['args'] = args

    logs_fname = f'logs/model_{args.dict}/phase1_rbc_{args.logs_fname}_grid_pi'

    # load data
    data_path = 'data/nse_data_reg_rbc'
    tg = args.tg     # sample evrey 5 timestamps
    Ng = args.Ng
    obs, temp , ctr = torch.load(data_path)
    end = 40

    def choose_data(data_list, Ng, end):
        data_chosen = []
        for data in data_list:
            length = int(data.shape[0]//4)
            data = data[length: length * 3 + 1, :-end + 1]
            data = data[::Ng]
            data_chosen.append(data)
        return data_chosen

    # obs = obs[::Ng, :-end + 1]
    # temp = temp[::Ng, :-end + 1]
    # ctr = ctr[::Ng, :-end + 1]
    obs, temp, ctr = choose_data([obs, temp, ctr], Ng, end)
    print('obs: ', obs.shape)
    print('temp: ', temp.shape)
    print('ctr: ', ctr.shape)

    N0, nt, nx, ny = obs.shape[0], obs.shape[1]-1, obs.shape[2], obs.shape[3]

    # logs['data_norm'] = data.normalize('unif')   # unif: min, range  norm: mean, var
    logs['pred_model'] = []
    logs['phys_model'] = []

    logs['test_loss_trans']=[]
    logs['test_loss_u_t_rec']=[]
    logs['test_loss_ctr_t_rec']=[]
    logs['test_loss_trans_latent']=[]
    logs['test_loss_pde_obs'] = []
    logs['test_loss_pde_pred'] = []

    # data param
    shape = [nx, ny]

    # loader
    class RBC_Dataset(Dataset):
        def __init__(self, obs, ctr):
            N0, nt, nx, ny = obs.shape[0], obs.shape[1]-1, obs.shape[2], obs.shape[3]
            self.Ndata = N0 * nt
            ctr = ctr[:, :-1]
            ctr = ctr.reshape(N0, nt, 1, 1, 1).repeat([1, 1, nx, ny, 1]).reshape(-1, nx, ny, 1)
            input_data = obs[:, :-1].reshape(-1, nx, ny, 3)
            output_data = obs[:, 1:].reshape(-1, nx, ny, 3)     #- input_data

            self.ipt = torch.cat((input_data, ctr), dim=-1)
            self.opt = output_data
            
        def __len__(self):
            return self.Ndata

        def __getitem__(self, idx):
            x = torch.FloatTensor(self.ipt[idx])
            y = torch.FloatTensor(self.opt[idx])
            return x, y
    
    RBC_data = RBC_Dataset(obs, ctr)
    tr_num = int(args.data_rate * RBC_data.Ndata)
    ts_num = int(0.2 * RBC_data.Ndata)
    train_data, test_data, _ = random_split(RBC_data, [tr_num, ts_num, RBC_data.Ndata - tr_num - ts_num])
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # model setting
    nse_model = RBCModel_FNO_prev(shape, 0.05, args)
    params_num = nse_model.count_params()

    print('N0: {}, nt: {}, nx: {}, ny: {}, device: {}'.format(N0, nt, nx, ny, nse_model.device))
    # print(f'ctr: {logs["data_norm"]["ctr"]}')
    # print(f'obs: {logs["data_norm"]["obs"]}')
    print(f'param numbers of the model: {params_num}')

    # train process
    for epoch in range(1, nse_model.params.epochs+1):
        nse_model.data_train(epoch, train_loader)
        # if epoch % nse_model.params.phys_gap == 0 and epoch != nse_model.params.epochs:
        #     # freeze phys_model trained in data training
        #     for param in list(nse_model.phys_model.parameters()):
        #         param.requires_grad = False
        #     for phys_epoch in range(1, nse_model.params.phys_epochs+1):
        #         nse_model.phys_train(phys_epoch, train_loader, random=args.phys_random_select)          
        #     for param in list(nse_model.phys_model.parameters()):
        #         param.requires_grad = True
        if epoch % 5 == 0:
            nse_model.save_log(logs)
            nse_model.test(test_loader, logs)

    torch.save([nse_model.pred_model.state_dict(), nse_model.phys_model.state_dict(), logs], logs_fname)
