import sys
sys.path.append("..")
sys.path.append("../env")

from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import matplotlib.pyplot as plt 
import torch
from fenics import * 
from timeit import default_timer

from scripts.nets import *
from scripts.utils import *

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

class DrawPlots:
    def __init__(self, num):
        fig, ax = plt.subplots(nrows=num, ncols=1, figsize=(12,10), dpi=1000)
        ax.flatten()

        for i in range(num):
            ax[i].grid(True, lw=0.4, ls="--", c=".50")
        
        self.num = num
        self.ax = ax

    def add_plot(self, t, data, t_start, label=None):
        for i in range(self.num):
            self.ax[i].plot(t[t_start:], data[i][t_start:], label=label)

    def save_fig(self, path):
        plt.savefig(path)

    def add_legend(self):
        for i in range(self.num):
            self.ax[i].legend()

    def add_ylabel(self, ylabel):
        for i in range(self.num):
            self.ax[i].set_ylabel(ylabel[i])
    