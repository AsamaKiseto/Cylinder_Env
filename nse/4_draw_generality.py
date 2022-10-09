# import torch
# import numpy as np
# import matplotlib.pyplot as plt
import argparse
from scripts.draw_utils import *

if __name__ == '__main__':
    
    nt = 80
    tg = 5
    t_nn = (np.arange(nt) + 1) * 0.01 * tg

    ex_nums = ['ex0', 'ex1_3', 'ex4_3']
    label = ['baseline', '2-step', '1-step']
    
    name_ex = '1_2'
    logs = torch.load('logs/phase1_env_logs_' + name_ex)

    scale_k = [0, 5, 10]
    # draw_generality(logs, ex_nums, label, tl)
    for i in range(len(scale_k)):
        draw_generality_multi(logs, ex_nums, label, t_nn, scale_k[i], name_ex)