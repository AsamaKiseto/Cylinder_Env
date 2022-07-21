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
    parser.add_argument('--data_path', default='./data/nse_control_samples', type=str, help='path of control data')
    parser.add_argument('--name', default='nse_control_p', type=str, help='experiment name')
    parser.add_argument('--data_num', default=0, type=int, help='data number')
    
    parser.add_argument('--L', default=4, type=int, help='the number of layers')
    parser.add_argument('--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('--width', default=20, type=int, help='the number of width of FNO layer')
        
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    parser.add_argument('--epochs', default=1000, type=int, help='number of Epochs')
    parser.add_argument('--lr', default=5e-1, type=float, help='learning rate')
    parser.add_argument('--step_size', default=200, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.8, type=float, help='scheduler factor')

    return parser.parse_args(argv)

# env init
env = Cylinder_Rotation_Env(params={'dtr': 0.01, 'T': 2, 'rho_0': 1, 'mu' : 1/1000,
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
    ftext = open('logs/nse_control_p.txt', mode="a", encoding="utf-8")
    ftext.write(f"{args.name} | data_num: {args.data_num}")

    # param setting
    if args.gpu==-1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(args.gpu))
    operator_path = args.operator_path
    data_path = args.data_path
    data_num = args.data_num
    L = args.L
    modes = args.modes
    width = args.width
    lr = args.lr
    epochs = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    
    # load_data
    data_path = './data/nse_data_N0_25_dtr_0.01_T_2'
    data_num = 0
    data, _, Cd, Cl, ang_vel = torch.load(data_path, map_location=lambda storage, loc: storage)
    data_in = data[data_num].squeeze()[0].to(device)
    ang_in = ang_vel[data_num][0]
    print('ang: {}'.format(ang_vel[data_num]))
    data_fin = data[data_num].squeeze()[0].to(device)

    # data params
    ny = data.shape[2] 
    nx = data.shape[3]
    s = data.shape[2] * data.shape[3]     # ny * nx
    N0 = data.shape[0]                    # num of data sets
    nt = data.shape[1] - 1                # nt
    print('N0: {}, nt: {}, ny: {}, nx: {}'.format(N0, nt, ny, nx))
    nt = 10

    # load model
    load_model = FNO(modes, modes, width, L).to(device)
    state_dict = torch.load(operator_path)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False
    
    # set policy net
    # p_net = torch.nn.ModuleList([policy_net_cnn().to(device) for _ in range(nt)])
    p_net = policy_net_cnn().to(device)
    
    # training
    optimizer = torch.optim.Adam(p_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(1, epochs + 1):
        print('start epoch {}'.format(epoch))
        env.reset()
        out_obs, _, Cd, Cl = env.step(ang_in.to(torch.device('cpu')).detach().numpy())
        print("ang_in: {}, Cd: {}, Cl: {}".format(ang_in.item(), Cd, Cl))

        p_net.train()
        optimizer.zero_grad()

        ang_optim = torch.rand(nt).to(device)
        out_nn = data_in.reshape(ny, nx, 3).to(device)
        Cd_nn = torch.zeros(nt).to(device)
        Cl_nn = torch.zeros(nt).to(device)
        Cd_obs = torch.zeros(nt).to(device)
        Cl_obs = torch.zeros(nt).to(device)
        for i in range(nt):
            # print('ang_optim[i]: {}'.format(ang_optim[i].size()))
            # ang_optim[i] = p_net(out_nn.reshape(1, -1))
            ang_optim[i] = p_net(out_nn.reshape(1, ny, nx, 3)) 
            ang_nn = ang_optim[i].reshape(1, 1, 1).repeat(ny, nx, 1)
            in_nn = torch.cat((out_nn.squeeze(), ang_nn), dim=-1).unsqueeze(0)
            out_nn, Cd_nn[i], Cl_nn[i] = load_model(in_nn)
            ang_obs = ang_optim[i].to(torch.device('cpu')).detach().numpy()
            out_obs, _, Cd_obs[i], Cl_obs[i] = env.step(ang_obs)
            print(ang_optim[i].item(), Cd_nn[i].item(), Cd_obs[i].item(), Cl_nn[i].item(), Cl_obs[i].item())
            # print(f"epoch: {epoch} | Cd_nn: {Cd_nn} | Cl_nn: {Cl_nn} | i: {i}")

        loss = torch.mean(Cd_nn ** 2) + 0.1 * torch.mean(Cl_nn ** 2)
        loss += torch.mean(ang_optim.squeeze() ** 2)
        print("epoch: {:4}  loss: {:1.6f}  Cd_nn: {:1.4f}  Cd_obs: {:1.4f}  Cd_mse: {:1.4f}  Cl_nn: {:1.4f}  Cl_obs: {:1.4f}  Cd_mse: {:1.4f}  ang_optim: {:1.3f}"
              .format(epoch, loss, Cd_nn.mean(), Cd_obs.mean(), ((Cd_nn-Cd_obs)**2).mean(), Cl_nn.mean(), Cl_obs.mean(), ((Cl_nn-Cl_obs)**2).mean(), ang_optim.mean()))
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # save log
        ftext.write("epoch: {:4}  loss: {:1.6f}  Cd_nn: {:1.6f}  Cd_obs: {:1.6f}  Cl_nn: {:1.6f}  Cl_obs: {:1.6f}  ang_optim: {}"
                     .format(epoch, loss, Cd_nn.mean(), Cd_obs.mean(), Cl_nn.mean(), Cl_obs.mean(), ang_optim))
    ftext.close()
