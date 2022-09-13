# import torch
# import numpy as np
# import matplotlib.pyplot as plt
import argparse
from scripts.draw_utils import *

def get_args(argv=None):
    parser = argparse.ArgumentParser(description='Put your hyperparameters')
    parser.add_argument('-tl', '--t_length', default=40, type=int, help='length of t for accumulate errors')
    return parser.parse_args(argv)

if __name__ == '__main__':
    # argparser
    args = get_args()
    tl = args.t_length

    ex_nums = ['ex0', 'ex8']
    # ex_nums = ['ex0_big', 'ex3_big', 'ex3_big_nomod']
    label = ['without_pde_loss', 'with_modify']

    logs = torch.load('logs/phase1_env_logs')

    draw_generality(logs, ex_nums, label, tl)