import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *

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
    parser.add_argument('--num_hiddens', type=list, default=[256, 2048, 1024, 1024, 128], metavar='N', help='number of hidden nodes of DDP_net')    
    
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    parser.add_argument('--epochs', default=1000, type=int, help='number of Epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--step_size', default=200, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')

    return parser.parse_args(argv)


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
    L = args.L
    modes = args.modes
    width = args.width
    lr = args.lr
    epochs = args.epochs
    step_size = args.step_size
    gamma = args.gamma
    
    # load_data
    data, Cd, Cl, _, ang_vel = torch.load(data_path, map_location=lambda storage, loc: storage)
    data_in = data[args.data_num].squeeze()[0].to(device)
    data_fin = data[args.data_num].squeeze()[0].to(device)

    # data params
    ny = data.shape[2] 
    nx = data.shape[3]
    s = data.shape[2] * data.shape[3]     # ny * nx
    N0 = data.shape[0]                    # num of data sets
    # nt = data.shape[1] - 1                # nt
    nt = 20
    print('N0: {}, nt: {}, ny: {}, nx: {}'.format(N0, nt, ny, nx))

    # load model
    load_model = FNO(modes, modes, width, L).to(device)
    state_dict, _ = torch.load(operator_path)
    load_model.load_state_dict(state_dict)
    load_model.eval()

    for param in list(load_model.parameters()):
        param.requires_grad = False
    
    # set policy net
    num_hiddens = [3 * s, 1024, 1024, 128, 1]       # last dim nt
    p_net = torch.nn.ModuleList([policy_net(activate=nn.Tanh(), num_hiddens=num_hiddens).to(device) for _ in range(nt)])
    # p_net = policy_net(activate=nn.Tanh(), num_hiddens=num_hiddens).to(device)
    # p_net = policy_fno_net(8, 8, 16, 3).to(device)
    
    # training
    optimizer = torch.optim.Adam(p_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    for epoch in range(1, epochs + 1):

        p_net.train()
        optimizer.zero_grad()

        ang_optim = torch.rand(nt).to(device)
        out_nn = data_in.reshape(ny, nx, 3).to(device)
        Cd_nn = torch.zeros(nt).to(device)
        Cl_nn = torch.zeros(nt).to(device)
        for i in range(nt):
            # print('ang_optim[i]: {}'.format(ang_optim[i].size()))
            # ang_optim[i] = p_net(out_nn.reshape(1, -1))
            ang_optim[i] = p_net[i](out_nn.reshape(1, -1))
            # ang_optim[i] = p_net(out_nn.reshape(1, ny, nx, 3))
            # print(ang_optim[i].size(), ang_optim[i])
            ang_nn = ang_optim[i].reshape(1, 1, 1).repeat(ny, nx, 1)
            in_nn = torch.cat((out_nn.squeeze(), ang_nn), dim=-1).unsqueeze(0)
            out_nn, Cd_nn[i], Cl_nn[i] = load_model(in_nn)
            # print(f"epoch: {epoch} | Cd_nn: {Cd_nn} | Cl_nn: {Cl_nn} | i: {i}")
        
    
        loss = torch.mean(Cd_nn ** 2) + 0.1 * torch.mean(Cl_nn ** 2)
        # loss += 0.01 * torch.sum(ang_optim.squeeze() ** 2)
        print("epoch: {:4}    loss: {:1.6f}    Cd_nn: {:1.6f}    Cl_nn: {:1.6f}    ang_optim: {:1.6f}"
              .format(epoch, loss, Cd_nn.mean(), Cl_nn.mean(), ang_optim.mean()))

        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        # save log
        ftext.write("epoch: {:3}    loss: {:1.6f}    Cd_nn: {:1.6f}    Cl_nn: {:1.6f}    ang_optim: {:1.6f}\n"
                    .format(epoch, loss, Cd_nn.mean(), Cl_nn.mean(), ang_optim.mean()))
    ftext.close()
