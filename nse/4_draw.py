from scripts.draw_utils import *

# log_list = ['data_based', 'ps_0.1']
# loss_plot(log_list)

tg = 5
t_nn = (np.arange(80) + 1) * 0.01 * tg   ### %%%
scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

scale_k = [0]

print('begin plot phys data num')
log_list = ['dr_0.3', 'dr_0.5', 'baseline']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_dr')

print('begin plot phys scale')
log_list = ['ps_0.01', 'ps_0.03', 'ps_0.05', 'baseline', 'ps_0.2', 'ps_0.3']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps')

print('begin plot phys train')
log_list = ['baseline', 'no_random', 'random_select']
test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pt')

print('begin plot diff t_start')
log_list = ['data_based', 'baseline']
ts_list = ['0.5', '1.0', '1.5', '2.0']
test_plot1(t_nn, log_list, scale_k, ts_list, ex_name='fb_0.0', fig_name='f_ts')

# print('begin plot')
# log_list = ['baseline', 'pe_5', 'pe_15', 'pe_20']
# test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe')

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
