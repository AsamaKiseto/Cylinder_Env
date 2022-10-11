from scripts.draw_utils import *

log_list = ['data_based']
t_nn = (np.arange(80) + 1) * 0.01 * 5   ### %%%

loss_plot(log_list)
for i in [0, 4, 9]:
    test_plot(log_list, t_nn, i)