from ast import operator
import sys
sys.path.append("..")
sys.path.append("../env")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    
    parser.add_argument('--operator_path', default='./logs/nse_operator_fno', type=str, help='path of operator weight')
    parser.add_argument('--data_num', default=0, type=int, help='data number')
    
    parser.add_argument('--L', default=2, type=int, help='the number of layers')
    parser.add_argument('--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('--width', default=20, type=int, help='the number of width of FNO layer')
    
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    parser.add_argument('--epochs', default=500, type=int, help='number of Epochs')
    parser.add_argument('--lr', default=5e-1, type=float, help='learning rate')

    return parser.parse_args(argv)


# env init
env = Cylinder_Rotation_Env(params={'dtr': 0.01, 'T': 1, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 128, 'dimy': 64,
                                    'min_x' : 0,  'max_x' : 2.2, 
                                    'min_y' : 0,  'max_y' : 0.41, 
                                    'r' : 0.05,  'center':(0.2, 0.2),
                                    'min_w': -1, 'max_w': 1,
                                    'min_velocity': -1, 'max_velocity': 1,
                                    'U_max': 1.5, })

print(env.params)


if __name__ == '__main__':
    # argparser
    args = get_args()

    # log text
    ftext = open('logs/nse_control_fno.txt', mode="a", encoding="utf-8")

    # param setting
    if args.gpu==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    operator_path = args.operator_path
    L = args.L
    modes = args.modes
    width = args.width
    lr = args.lr
    epochs = args.epochs

    model_params = dict()
    model_params['modes'] = args.modes
    model_params['width'] = args.width
    model_params['L'] = args.L

    f_channels = 4
    
    # load_data
    data_path = './data/nse_data_N0_256_nT_400'
    data_num = 0
    data, _, Cd, Cl, ang_vel = torch.load(data_path, map_location=lambda storage, loc: storage)
    print('load data finished')
    tg = 20     # sample evrey 10 timestamps
    Ng = 1
    data = data[::Ng, ::tg, :, :, 2:]  
    Cd = Cd[::Ng, ::tg]
    Cl = Cl[::Ng, ::tg]
    ang_vel = ang_vel[::Ng, ::tg]

    ang_in = ang_vel[data_num][0]
    # print('ang: {}'.format(ang_vel[data_num]))
    data_in = data[data_num].squeeze()[0].to(device)

    # data params
    nx = data.shape[2] 
    ny = data.shape[3]
    shape = [nx, ny]
    # s = data.shape[2] * data.shape[3]     # ny * nx
    N0 = data.shape[0]                    # num of data sets
    nt = data.shape[1] - 1                # nt
    Ndata = N0 * nt
    print('N0: {}, nt: {}, nx: {}, ny: {}'.format(N0, nt, nx, ny))

    # load_model
    operator_path = 'logs/operator_lambda_1_0.1_0.1_0.1_lr_0.001_epochs_2000'
    load_model = FNO_ensemble(model_params, shape, f_channels=f_channels).to(device)
    state_dict, _ = torch.load(operator_path)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False

    # training
    ang_optim = torch.rand(nt).to(device)
    ang_optim.requires_grad = True
    print('ang_optim: {}'.format(ang_optim.size()))
    
    # optimizer = torch.optim.LBFGS([ang_optim], lr=lr, line_search_fn='strong_wolfe')
    optimizer = torch.optim.Adam([ang_optim], lr=lr)
    
    for epoch in range(1, epochs + 1):
        env.reset()
        # env.step(ang_in.to(torch.device('cpu')).detach().numpy())
        optimizer.zero_grad()

        out_nn = data_in.reshape(1, nx, ny, 3).to(device)
        f_rec = torch.zeros(nt).to(device)
        Cd_nn = torch.zeros(nt).to(device)
        Cl_nn = torch.zeros(nt).to(device)
        Cd_obs = torch.zeros(nt).to(device)
        Cl_obs = torch.zeros(nt).to(device)
        
        loss = 0
        for i in range(nt):
            ang_nn = ang_optim[i].reshape(1)
            pred, _, f_rec[i], _ = load_model(out_nn, ang_nn)
            out_nn = pred[:, :, :, :3]
            Cd_nn[i] = torch.mean(pred[:, :, :, -2])
            Cl_nn[i] = torch.mean(pred[:, :, :, -1])
            ang_obs = ang_optim[i].to(torch.device('cpu')).detach().numpy()
            for j in range(tg-1):
                env.step(ang_obs)
            out_obs, _, Cd_obs[i], Cl_obs[i] = env.step(ang_obs)
            print(ang_optim[i].item(), Cd_nn[i].item(), Cd_obs[i].item(), Cl_nn[i].item(), Cl_obs[i].item())
            # print(f"epoch: {epoch} | Cd_nn: {Cd_nn} | Cl_nn: {Cl_nn} | i: {i}")
        
        loss = torch.mean(Cd_nn ** 2) + 0.1 * torch.mean(Cl_nn ** 2)
        loss += torch.mean((ang_optim - f_rec) ** 2)
        # loss += 0.001 * torch.mean(ang_optim.squeeze() ** 2)
        print("epoch: {:4}  loss: {:1.6f}  Cd_nn: {:1.6f}  Cd_obs: {:1.6f}  Cl_nn: {:1.6f}  Cl_obs: {:1.6f}  ang_optim: {:1.6f}"
              .format(epoch, loss, Cd_nn.mean(), Cd_obs.mean(), Cl_nn.mean(), Cl_obs.mean(), ang_optim.mean()))
        loss.backward()
        optimizer.step()
        
        # save log
        ftext.write("epoch: {:4}  loss: {:1.6f}  Cd_nn: {:1.6f}  Cd_obs: {:1.6f}  Cl_nn: {:1.6f}  Cl_obs: {:1.6f}  ang_optim: {}"
                    .format(epoch, loss, Cd_nn.mean(), Cd_obs.mean(), Cl_nn.mean(), Cl_obs.mean(), ang_optim))
    ftext.close()