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
    
    parser.add_argument('--operator_path', default='phase1_ex12_norm', type=str, help='path of operator weight')
    parser.add_argument('--data_num', default=0, type=int, help='data number')
    parser.add_argument('--t_start', default=0, type=int, help='data number')
    
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    parser.add_argument('--epochs', default=500, type=int, help='number of Epochs')
    parser.add_argument('--lr', default=5e-1, type=float, help='learning rate')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')

    return parser.parse_args(argv)


# env init
# env = Cylinder_Rotation_Env(params={'dtr': 0.01, 'T': 1, 'rho_0': 1, 'mu' : 1/1000,
#                                     'traj_max_T': 20, 'dimx': 128, 'dimy': 64,
#                                     'min_x' : 0,  'max_x' : 2.2, 
#                                     'min_y' : 0,  'max_y' : 0.41, 
#                                     'r' : 0.05,  'center':(0.2, 0.2),
#                                     'min_w': -1, 'max_w': 1,
#                                     'min_velocity': -1, 'max_velocity': 1,
#                                     'U_max': 1.5, })

# print(env.params)


if __name__ == '__main__':
    # argparser
    args = get_args()

    # path & load
    data_path = './data/nse_data_N0_256_nT_400'
    operator_path = './logs/' + args.operator_path

    data_orig, _, Cd, Cl, ang_vel = torch.load(data_path, map_location=lambda storage, loc: storage)
    state_dict, logs_model = torch.load(operator_path)

    # log text
    ftext = open('logs/nse_control_fno.txt', mode="a", encoding="utf-8")
    logs = dict()
    logs['operator_path'] = operator_path
    logs['obs_nn'] = []
    logs['Cd_nn'] = []
    logs['Cl_nn'] = []
    logs['loss'] = []

    # param setting
    if args.gpu==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    epochs = args.epochs
    lr = args.lr
    step_size = args.step_size
    gamma = args.gamma

    L = logs_model['args'].L
    modes = logs_model['args'].modes
    width = logs_model['args'].width
    model_params = dict()
    model_params['modes'] = modes
    model_params['width'] = width
    model_params['L'] = L

    f_channels = logs_model['args'].f_channels
    
    # data setting
    data_num = args.data_num
    t_start = args.t_start
    logs['data_num'] = data_num
    logs['t_start'] = t_start
    
    Cd_mean, Cd_var = logs_model['data_norm']['Cd']
    Cl_mean, Cl_var = logs_model['data_norm']['Cl']
    ang_vel_mean, ang_vel_var = logs_model['data_norm']['f']
    Cd_mean, Cd_var = Cd_mean[0], Cd_var[0]
    Cl_mean, Cl_var = Cl_mean[0], Cl_var[0]
    
    print('load data finished')
    tg = logs_model['args'].tg     # sample evrey 10 timestamps
    Ng = logs_model['args'].Ng
    data = data_orig[::Ng, ::tg, :, :, 2:]  
    Cd = Cd[::Ng, ::tg]
    Cl = Cl[::Ng, ::tg]
    ang_vel = ang_vel[::Ng, ::tg]

    print(f'ang_vel: {ang_vel[data_num]}')
    data_in = data[data_num].squeeze()[t_start].to(device)

    # data params
    nx = data.shape[2] 
    ny = data.shape[3]
    shape = [nx, ny]
    N0 = data.shape[0]                    # num of data sets
    nt = data.shape[1] - 1                # nt
    Ndata = N0 * nt
    print('N0: {}, nt: {}, nx: {}, ny: {}'.format(N0, nt, nx, ny))

    # load_model
    load_model = FNO_ensemble(model_params, shape, f_channels=f_channels).to(device)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False

    # training
    obs_nn = torch.zeros(nt, nx, ny, 3)
    obs_nn[:t_start] = data[data_num, 1:t_start+1]
    # ang_optim = torch.rand(nt).to(device)
    # ang_optim[:t_start] = ang_vel[data_num][:t_start]
    ang_optim = ang_vel[data_num].to(device)
    print(ang_optim.shape)
    ang_optim.requires_grad = True
    
    optimizer = torch.optim.Adam([ang_optim], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(1, epochs + 1):
        # env.reset()
        optimizer.zero_grad()

        out_nn = data_in.reshape(1, nx, ny, 3).to(device)
        f_rec = torch.zeros(nt).to(device)
        Cd_nn = torch.zeros(nt).to(device)
        Cl_nn = torch.zeros(nt).to(device)
        Cd_obs = torch.zeros(nt).to(device)
        Cl_obs = torch.zeros(nt).to(device)
        
        loss = 0
        for i in range(t_start, nt):
            ang_nn = ang_optim[i].reshape(1)
            # ang_nn = ang_vel[data_num][i].reshape(1)
            # print(ang_nn.shape)
            pred, _, f_rec[i], _ = load_model(out_nn, ang_nn)
            out_nn = pred[:, :, :, :3]
            obs_nn[i] = out_nn
            Cd_nn[i] = torch.mean(pred[:, :, :, -2]) 
            Cl_nn[i] = torch.mean(pred[:, :, :, -1]) 
            # ang_obs = ang_optim[i].to(torch.device('cpu')).detach().numpy()
            # for j in range(tg-1):
            #     env.step(ang_obs)
            # out_obs, _, Cd_obs[i], Cl_obs[i] = env.step(ang_obs)
            # print(ang_optim[i].item(), Cd_nn[i].item(), Cd_obs[i].item(), Cl_nn[i].item(), Cl_obs[i].item())
        
        Cd_nn = Cd_nn * Cd_var.to(device) + Cd_mean.to(device)
        Cl_nn = Cl_nn * Cl_var.to(device) + Cl_mean.to(device)
        loss = torch.mean(Cd_nn[t_start:] ** 2) + 0.1 * torch.mean(Cl_nn[t_start:] ** 2)
        loss += 0.05 * torch.mean((ang_optim[t_start:] - f_rec[t_start:]) ** 2)
        # loss += 0.001 * torch.mean(ang_optim.squeeze() ** 2)
        if(epoch%10 == 0):
            print("epoch: {:4}  loss: {:1.6f}  Cd_nn: {:1.6f}  Cd_obs: {:1.6f}  Cl_nn: {:1.6f}  Cl_obs: {:1.6f}  ang_optim: {:1.6f}"
                  .format(epoch, loss, Cd_nn[t_start:].mean(), Cd_obs[t_start:].mean(), Cl_nn[t_start:].mean(), Cl_obs[t_start:].mean(), ang_optim[t_start:].mean()))
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        logs['loss'].append(loss)
        logs['obs_nn'].append(obs_nn)
        logs['Cd_nn'].append(Cd_nn)
        logs['Cl_nn'].append(Cl_nn)
        # save log
        ftext.write("epoch: {:4}  loss: {:1.6f}  Cd_nn: {:1.6f}  Cd_obs: {:1.6f}  Cl_nn: {:1.6f}  Cl_obs: {:1.6f}  ang_optim: {}\n"
                    .format(epoch, loss, Cd_nn[t_start:].mean(), Cd_obs[t_start:].mean(), Cl_nn[t_start:].mean(), Cl_obs[t_start:].mean(), ang_optim[t_start:]))
    ftext.close()

    print(ang_optim)
    logs['f_optim'] = ang_optim
    torch.save(logs, 'logs/phase2_logs_test')