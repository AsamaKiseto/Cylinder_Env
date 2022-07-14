import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm, trange
from timeit import default_timer
import sys
import os

import argparse

from models import FNO
from utils import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')

    parser.add_argument('--name', default='nse_operator_fno', type=str, help='experiments name')
    
    parser.add_argument('--L', default=4, type=int, help='the number of layers')
    parser.add_argument('--modes', default=12, type=int, help='the number of modes of Fourier layer')
    parser.add_argument('--width', default=20, type=int, help='the number of width of FNO layer')
    
    parser.add_argument('--batch', default=20, type=int, help = 'batch size')
    parser.add_argument('--epochs', default=1000, type=int, help = 'Number of Epochs')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--step_size', default=100, type=int, help='scheduler step size')
    parser.add_argument('--gamma', default=0.5, type=float, help='scheduler factor')
    parser.add_argument('--weight', default=1.0, type=float, help='weight of recon loss')
    parser.add_argument('--gpu', default=0, type=int, help='device number')
    
    return parser.parse_args(argv)

if __name__=='__main__':
    # args parser
    args = get_args()
    print(args)
    
    # output
    ftext = open('./logs/nse_operator_fno.txt', 'w', encoding='utf-8')
    
    # param setting
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    L = args.L
    modes = args.modes
    width = args.width
    
    epochs = args.epochs
    batch_size = args.batch
    lr = args.lr
    wd = args.wd
    step_size = args.step_size
    gamma = args.gamma
    # weight = args.weight
    
    fname = './logs/{}'.format(args.name)
        
    # load data
    data, _, Cd, Cl, ang_vel = torch.load('data/nse_data_N0_25_dtr_0.05_T_5')
    Cd = Cd[:, 1:]
    Cl = Cl[:, 1:]
    ang_vel = ang_vel[:, 1:]
    # u_data = data[:, :, :, :, :-1]
    # p_data = data[:, :, :, :, -1]

    # data param
    ny = data.shape[2] 
    nx = data.shape[3]
    s = data.shape[2] * data.shape[3]     # ny * nx
    N0 = data.shape[0]                    # num of data sets
    nt = data.shape[1] - 1             # nt
    Ndata = N0 * nt
    
    print('N0: {}, nt: {}, ny: {}, nx: {}'.format(N0, nt, ny, nx))
    
    class NSE_Dataset(Dataset):
        def __init__(self, data, Cd, Cl, ang_vel):
            Cd = Cd.reshape(N0, nt, 1, 1, 1).repeat([1, 1, ny, nx, 1]).reshape(-1, ny, nx, 1)
            Cl = Cl.reshape(N0, nt, 1, 1, 1).repeat([1, 1, ny, nx, 1]).reshape(-1, ny, nx, 1)
            ang_vel = ang_vel.reshape(N0, nt, 1, 1, 1).repeat([1, 1, ny, nx, 1]).reshape(-1, ny, nx, 1)
            input_data = data[:, :-1].reshape(-1, ny, nx, 3)
            output_data = data[:, :-1].reshape(-1, ny, nx, 3)

            self.input_data = torch.cat((input_data, ang_vel), dim=-1)
            self.output_data = torch.cat((output_data, Cd, Cl), dim=-1)
            
        def __len__(self):
            return Ndata

        def __getitem__(self, idx):
            x = torch.FloatTensor(self.input_data[idx])
            y = torch.FloatTensor(self.output_data[idx])
            return x, y

    NSE_data = NSE_Dataset(data, Cd, Cl, ang_vel)
    train_data, test_data = random_split(NSE_data, [int(0.8 * Ndata), int(0.2 * Ndata)])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    # model setting
    model = FNO(modes, modes, width, L).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    logs = dict()
    logs['epoch_time'] = []
    logs['train_loss'] = []
    logs['train_mse'] = []
    logs['test_loss'] = []

    pbar = tqdm(total=epochs, file=sys.stdout)
    for epoch in range(1, epochs+1):
        model.train()
        
        t1 = default_timer()
        train_loss1 = AverageMeter()
        train_loss2 = AverageMeter()
        test_loss1 = AverageMeter()
        test_loss2 = AverageMeter()
        train_mse = AverageMeter()

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            optimizer.zero_grad()

            out_train, Cd_train, Cl_train = y_train[:, :, :, :3], y_train[:, 0, 0, 3], y_train[:, 0, 0, 4]
            out_pred, Cd_pred, Cl_pred = model(x_train)
            loss1 = rel_error(out_pred, out_train).mean()
            loss2 = F.mse_loss(Cd_train, Cd_pred, reduction='mean') + F.mse_loss(Cl_train, Cl_pred, reduction='mean')
            # loss2 = (Cd_train - Cd_pred) ** 2 + (Cl_train - Cl_pred) ** 2
            mse = F.mse_loss(out_pred.reshape(batch_size, -1), out_train.reshape(batch_size, -1), reduction='mean')
            loss = loss1 + loss2
            loss.backward()

            optimizer.step()

            train_loss1.update(loss1.item(), x_train.shape[0])
            train_loss2.update(loss2.item(), x_train.shape[0])
            train_mse.update(mse.item(), x_train.shape[0])
        
        scheduler.step()

        model.eval()
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)

                out_test, Cd_test, Cl_test = y_test[:, :, :, :3], y_test[:, 0, 0, 3], y_test[:, 0, 0, 4]
                out_pred, Cd_pred, Cl_pred = model(x_test)
                loss1 = rel_error(out_pred, out_test).mean()
                loss2 = F.mse_loss(Cd_test, Cd_pred, reduction='mean') + F.mse_loss(Cl_test, Cl_pred, reduction='mean')
                loss = loss1 + loss2

                test_loss1.update(loss1.item(), x_test.shape[0])
                test_loss2.update(loss2.item(), x_test.shape[0])
            
        t2 = default_timer()
        
        logs['epoch_time'].append(t2 - t1)
        logs['train_loss'].append(train_loss1.avg)
        logs['train_mse'].append(train_mse.avg)
        logs['test_loss'].append(test_loss1.avg)

        ftext.write('epoch {} | (time) epoch_time: {:1.4e} | (train) mse: {:1.4e},  loss1: {:1.4e},  loss2: {:1.4e} | (test) loss1: {:1.4e}, loss2: {:1.4e}'
                    .format(epoch, t2-t1, train_mse.avg, train_loss1.avg, train_loss2.avg, test_loss1.avg, test_loss2.avg))
        
        end = '\r'
        pbar.set_description('epoch {} | (time) epoch_time: {:1.4e} | (train) mse: {:1.4e},  loss1: {:1.4e},  loss2: {:1.4e} | (test) loss1: {:1.4e}, loss2: {:1.4e}'
                             .format(epoch, t2-t1, train_mse.avg, train_loss1.avg, train_loss2.avg, test_loss1.avg, test_loss2.avg))
        pbar.update()
        
    ftext.close()
    # if not os.path.exists('./logs/{}'.format(args.name)):
    #     os.mkdir('./logs/{}'.format(args.name))
    torch.save([model.state_dict(), logs], fname)
