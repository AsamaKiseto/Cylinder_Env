from scripts.test_utils import *
import matplotlib.pyplot as plt 
import argparse

def get_args(argv=None):
    parser = argparse.ArgumentParser(description = 'Put your hyperparameters')
    parser.add_argument('-fn', '--file_name', default='data_based', type=str, help='model path name')
    parser.add_argument('-tg', '--tg', default=5, type=int, help='model path name')

    return parser.parse_args(argv)

args = get_args()
file_name = args.file_name
tg = args.tg

# loss 
data_path = 'data/nse_data_reg_dt_0.01_fr_1.0'
data = LoadData(data_path)
data.split(1, tg)
N0, nt, nx, ny = data.get_params()
loss_log(data, file_name)

# test 
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadData(data_path)
data.split(1, tg)
test_log(data, file_name, 'fb_0.0')

data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_1.0'
data = LoadData(data_path)
data.split(1, tg)
test_log(data, file_name, 'fb_1.0')

