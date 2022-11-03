from scripts.draw_utils import *

# log_list = ['data_based', 'ps_0.1']
# loss_plot(log_list)

t_nn = (np.arange(80) + 1) * 0.05    ### %%%

scale_k = [2]

print('begin plot train method')
# log_list = ['data_based']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='data_based', dict = 'nse')

# log_list = ['phys_inc', 'data_based']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='train_method1', dict = 'nse')

# log_list = ['phys_inc', 'data_based', 'random_select_0.001', 'no_random']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='train_method2', dict = 'nse')

log_list = ['phys_inc', 'data_based']
test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0_1', fig_name='train_method', dict = 'nse', zlim=0.5)

# print('begin plot data num')
# # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_data')
# log_list = ['data_based', 'baseline', 'dr_0.3', 'dr_0.5', 'dr_0.3_0', 'dr_0.5_0']
# test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_data', dict = 'nse')

# print('begin plot phys scales')
# # # log_list = ['baseline', 'ps_0.01', 'ps_0.05', 'ps_0.2', 'ps_0.5']
# # # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps')
# log_list = ['baseline', 'ps_0.01', 'ps_0.05', 'ps_0.2', 'ps_0.3']
# test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps', dict = 'nse')

# print('begin plot phys epochs')
# log_list = ['baseline', 'pe_5', 'pe_15', 'pe_20']
# # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe')
# test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe', dict = 'nse')

# print('begin plot phys steps')
# log_list = ['baseline', 'psp_1', 'psp_3']
# # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_psp')
# test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_psp', dict = 'nse')

# test_data_name = 'fr_1.0'
# data_path = 'data/nse_data_reg_dt_0.01_' + test_data_name
# print('load data')
# data = LoadData(data_path)
# data.split(1, tg)
# print('load data finished')
# data.normalize()
# _, Cd, Cl, _ = data.get_data()
# coef_plot1(t_nn, [Cd, Cl], test_data_name)

# test_data_name = 'fb_0.0'
# data_path = 'data/test_data/nse_data_reg_dt_0.01_' + test_data_name
# print('load data')
# data = LoadData(data_path)
# data.split(1, tg)
# print('load data finished')
# data.normalize()
# _, Cd, Cl, _ = data.get_data()
# coef_plot(t_nn, scale_k, [Cd, Cl], test_data_name)




# t_nn = (np.arange(99) + 1) * 0.01   ### %%%

# scale_k = [2]

# print('begin plot train method')
# # log_list = ['data_based', 'baseline', 'no_random', 'random_select_0.01', 'random_select_0.001', 'random_select_0.0001', 'pre_phys']
# # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='train_method')
# log_list = ['data_based', 'phys_inc', 'no_random', 'random_select_0.001', 'prev_phys']
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='train_method', dict = 'rbc')

# print('begin plot phys scales')
# # # log_list = ['baseline', 'ps_0.01', 'ps_0.05', 'ps_0.2', 'ps_0.5']
# # # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps')
# log_list = ['phys_inc', 'ps_0.01', 'ps_0.05', 'ps_0.2']
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='f_ps', dict = 'rbc')

# print('begin plot phys epochs')
# log_list = ['phys_inc', 'pe_5', 'pe_15']
# # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe')
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='f_pe', dict = 'rbc')