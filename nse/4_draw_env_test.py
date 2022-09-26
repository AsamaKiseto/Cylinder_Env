import numpy as np
import matplotlib.pyplot as plt 
import torch

from scripts.models import *
from scripts.utils import *
from scripts.draw_utils import *

# load test data
data_path = 'data/test_data/nse_data_reg_scale...'
data = LoadData(data_path)
data.split(1, 5)
obs, Cd, Cl, ctr = data.get_data()
data_norm = data.norm()

N0, nt, nx, ny = data.get_params()
shape = [nx, ny]

tg = 5
nt = 80

t_nn = (np.arange(nt) + 1) * 0.01 * tg
t = (np.arange(nt * tg) + 1) * 0.01 

