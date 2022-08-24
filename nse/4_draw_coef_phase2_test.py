import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 

from scripts.models import *
from scripts.utils import *
from scripts.draw import *

# plot colors
from matplotlib import colors

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

import argparse

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    
    parser.add_argument('--k', default=200, type=int, help='data number')
    parser.add_argument('--t_start', default=0, type=int, help='data number')
    

    return parser.parse_args(argv)


if __name__ == '__main__':
    # argparser
    args = get_args()

    t_start = args.t_start
    logs = torch.load('logs/phase2_logs_test')
    operator_path = logs['operator_path']
    obs_nn = logs['obs_nn']
    Cd_nn = logs['Cd_nn']
    Cl_nn = logs['Cl_nn']
    data_num = logs['data_num']
    f_optim = logs['f_optim']
    # print(Cd_nn[-1])

    nt = Cd_nn[0].shape[0]
    Nk = 1
    epochs = 500
    k = (1 + np.arange(Nk))*(500//Nk) - 1

    data_path = './data/nse_data'
    data_orig, _, Cd, Cl, ang_vel = torch.load(data_path, map_location=lambda storage, loc: storage)
    
    _, logs_model = torch.load(operator_path)
    Cd_mean, Cd_var = logs_model['data_norm']['Cd']
    Cl_mean, Cl_var = logs_model['data_norm']['Cl']
    ctr_mean, ctr_var = logs_model['data_norm']['ctr']
    f = f_optim * ctr_var + ctr_mean
    print(f)

    print('load data finished')
    tg = logs_model['args'].tg     # sample evrey 10 timestamps
    Ng = logs_model['args'].Ng
    data = data_orig[::Ng, ::tg, :, :, 2:]
    Cd = Cd[::Ng, ::tg]
    Cl = Cl[::Ng, ::tg]
    ang_vel = ang_vel[::Ng, ::tg]
    # print(Cd[data_num])

    plots = DrawPlots(2)

    t_nn = np.arange(nt) * 0.01 * tg

    for i in k:
        Cd_ = Cd_nn[i].to(torch.device('cpu')) #* Cd_var + Cd_mean
        Cl_ = Cl_nn[i].to(torch.device('cpu')) #* Cl_var + Cl_mean
        Cd_ = Cd_.detach().numpy()
        Cl_ = Cl_.detach().numpy()

        plots.add_plot(t_nn, [Cd[data_num], Cl[data_num]], t_start)
        plots.add_plot(t_nn, [Cd_, Cl_], t_start, label=i+1)
    
    ylabel = [r"$Cd$", r"$Cl$"]
    plots.add_ylabel(ylabel)

    plots.save_fig('coef_phase2_test1.jpg')