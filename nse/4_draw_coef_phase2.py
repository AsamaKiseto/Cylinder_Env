import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 
from timeit import default_timer

from models import *
from utils import *

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
    
    parser.add_argument('--operator_path', default='phase1_logs_ex12', type=str, help='path of operator weight')
    parser.add_argument('--k', default=200, type=int, help='data number')
    parser.add_argument('--t_start', default=2, type=int, help='data number')
    

    return parser.parse_args(argv)

if __name__ == '__main__':
    # argparser
    args = get_args()

    t_start = args.t_start
    logs = torch.load('logs/phase2_logs_test')
    Cd_nn = logs['Cd_nn']
    Cl_nn = logs['Cl_nn']

    nt = Cd_nn[0].shape[0]
    Nk = 50
    k = np.arange(Nk)*(500//Nk)

    plt.figure(figsize=(15, 12))

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    t_nn = (np.arange(nt) + 1) * 0.01 * nt

    for i in range(Nk):
        Cd = Cd_nn[i].to(torch.device('cpu')).detach().numpy()
        Cl = Cl_nn[i].to(torch.device('cpu')).detach().numpy()

        ax1.plot(t_nn[t_start:], Cd[t_start:], color = cmap(i/(Nk+1)))
        ax1.grid(True, lw=0.4, ls="--", c=".50")
        ax1.set_ylabel(r"$Cd$", fontsize=15)
        ax1.set_xlim(0, 4)

        ax2.plot(t_nn[t_start:], Cl[t_start:], color = cmap(i/(Nk+1)))
        ax2.grid(True, lw=0.4, ls="--", c=".50")
        ax2.set_ylabel(r"$Cl$", fontsize=15)
        ax2.set_xlim(0, 4)

    plt.savefig(f'coef_phase2_#{k}_t_start_{t_start}.jpg')