from scripts.draw_utils import *

t_nn = (np.arange(80) + 1) * 0.01 * 5   ### %%%
log_list = ['data_based', 'ps_0.1']
# loss_plot(log_list)

scale_k = [0, 9]
log_list = ['data_based', 'ps_0.1']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='base')
test_plot(t_nn, log_list, scale_k, ex_name='fb_1.0', fig_name='base')

log_list = ['ps_0.01', 'baseline', 'ps_0.1']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps')
test_plot(t_nn, log_list, scale_k, ex_name='fb_1.0', fig_name='f_ps')

log_list = ['baseline', 'pe_20', 'pe_30']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe')
test_plot(t_nn, log_list, scale_k, ex_name='fb_1.0', fig_name='f_pe')
