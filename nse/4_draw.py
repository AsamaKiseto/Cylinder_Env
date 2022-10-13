from scripts.draw_utils import *

# log_list = ['data_based', 'ps_0.1']
# loss_plot(log_list)

tg = 5
t_nn = (np.arange(80) + 1) * 0.01 * tg   ### %%%
scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

scale_k = [0, 9]

print('begin plot')
log_list = ['data_based', 'ps_0.1']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='base')
test_plot(t_nn, log_list, scale_k, ex_name='fb_1.0', fig_name='base')

print('begin plot')
log_list = ['ps_0.01', 'baseline', 'ps_0.1']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps')
test_plot(t_nn, log_list, scale_k, ex_name='fb_1.0', fig_name='f_ps')

print('begin plot')
log_list = ['baseline', 'pe_20', 'pe_30']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe')
test_plot(t_nn, log_list, scale_k, ex_name='fb_1.0', fig_name='f_pe')

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

# test_data_name = 'fb_1.0'
# data_path = 'data/test_data/nse_data_reg_dt_0.01_' + test_data_name
# print('load data')
# data = LoadData(data_path)
# data.split(1, tg)
# print('load data finished')
# data.normalize()
# _, Cd, Cl, _ = data.get_data()
# coef_plot(t_nn, scale_k, [Cd, Cl], test_data_name)
