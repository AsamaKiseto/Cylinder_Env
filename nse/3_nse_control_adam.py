import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm, trange

import random
import os
import time

from models import *

import argparse


def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    
    parser.add_argument('--operator_path', default='./logs/nse_operator_fno', type=str, help='path of operator weight')
    parser.add_argument('--data_num', default=0, type=int, help='data number')
    
    parser.add_argument('--L', default=4, type=int, help='the number of layers')
    parser.add_argument('--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('--width', default=20, type=int, help='the number of width of FNO layer')
    
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    parser.add_argument('--epochs', default=500, type=int, help='number of Epochs')
    parser.add_argument('--lr', default=5e-1, type=float, help='learning rate')

    return parser.parse_args(argv)


if __name__ == '__main__':
    print('1111')
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
    
    # load_data
    data_path = './data/nse_data_N0_25_dtr_0.01_T_2'
    data_num = 0
    data, Cd, Cl, _, ang_vel = torch.load(data_path, map_location=lambda storage, loc: storage)
    data_in = data[data_num].squeeze()[0].to(device)
    data_fin = data[data_num].squeeze()[0].to(device)
    # print('data_in: {}'.format(data_in.size()))
    # print('data_fin: {}'.format(data_fin.size()))

    # data params
    ny = data.shape[2] 
    nx = data.shape[3]
    s = data.shape[2] * data.shape[3]     # ny * nx
    N0 = data.shape[0]                    # num of data sets
    nt = data.shape[1] - 1                # nt
    Ndata = N0 * nt
    print('N0: {}, nt: {}, ny: {}, nx: {}'.format(N0, nt, ny, nx))

    # load_model
    load_model = FNO(modes, modes, width, L).to(device)
    state_dict = torch.load(operator_path)
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
        optimizer.zero_grad()
        
        loss = 0
        out_nn = data_in.reshape(ny, nx, 3)
        tmp = []
        nt=20
        # print('out_nn: {}'.format(out_nn.size()))
        for i in range(nt):
            # print('ang_optim[i]: {}'.format(ang_optim[i].size()))
            this_f =  F.tanh(ang_optim[i]) * 0.5 + 0.5
            # print("ang_optim[i]: {}".format(ang_optim[i]))
            in_nn = torch.cat((out_nn.squeeze(), this_f.reshape(1, 1, 1).repeat(ny, nx, 1)), dim=-1).unsqueeze(0)
            # print('in_nn: {}'.format(in_nn.size()))
            out_nn, Cd_nn, Cl_nn = load_model(in_nn)
            # tmp.append(Cd_nn, Cl_nn, ang_optim[i])
            loss +=  Cd_nn ** 2 + 0.1 * Cl_nn ** 2
            tmp.append([Cd_nn.item(), Cl_nn.item(), loss.item(), ang_optim[i].item()])
            # print([Cd_nn.item(), Cl_nn.item(), loss.item(), ang_optim[i].item()])
        
        # loss += 0.01*torch.sum(ang_optim.squeeze() ** 2)
        tmp = np.array(tmp)
        print("epoch: {:4}    loss: {:1.6f}    Cd_nn: {:1.6f}    Cl_nn: {:1.6f}    ang_optim: {:1.6f}"
              .format(epoch, tmp.mean(0)[2], tmp.mean(0)[0], tmp.mean(0)[1], tmp.mean(0)[3]))
        ftext.write("epoch: {:4}    loss: {:1.6f}    Cd_nn: {:1.6f}    Cl_nn: {:1.6f}    ang_optim: {:1.6f}"
                    .format(epoch, tmp.mean(0)[2], tmp.mean(0)[0], tmp.mean(0)[1], tmp.mean(0)[3]))

        loss.backward() 
        optimizer.step()

    ftext.close()
