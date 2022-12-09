from scripts.draw_utils import *

# log_list = ['data_based', 'ps_0.1']
# loss_plot(log_list)

t_nn = (np.arange(80) + 1) * 0.05    ### %%%

scale_k = [0]

# print('begin plot train method')
# log_list = ['data_based']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='data_based', dict = 'nse')

log_list = ['phys_inc', 'data_based', 'trivial', 'test1']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='train_method1', dict = 'nse', label_list=['ours', 'data-based', 'trivial', 'test1'])

# log_list = ['phys_inc', 'data_based', 'random_select_0.001', 'no_random']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='train_method2', dict = 'nse', label_list=['ours', 'data-based', 'random select', 'no random'])

# log_list = ['phys_inc', 'data_based']
# test_plot1(t_nn, log_list, scale_k, ex_name='fb_0.0_scale_2.0', fig_name='train_method', dict = 'nse', zlim=0.5, label_list=['ours', 'data-based'])

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

t_nn = (np.arange(80) + 1) * 0.05   ### %%%
print('begin plot train method')
log_list = ['data_based', 'phys_inc', 'no_random', 'random_select_0.0001', 'ps_0.01', 'pe_5', 'pe_15', 'psp_1', 'psp_3']
test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='train_method1_0', dict = 'rbc')

t_nn = (np.arange(100) + 1) * 0.01   ### %%%

scale_k = [2]

print('begin plot train method')
log_list = ['data_based_1', 'phys_inc_1', 'no_random_1', 'random_select_1', 'random_select_0.0001_1', 'ps_0.1_1', 'ps_0.01_1', 'pe_5_1', 'pe_15_1']
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='train_method1_1', dict = 'rbc')


t_nn = (np.arange(80) + 1) * 0.05   ### %%%
print('begin plot train method')
log_list = ['data_based_2', 'phys_inc_2', 'no_random_2', 'random_select_2', 'random_select_0.0001_2', 'ps_0.1_2', 'ps_0.01_2', 'pe_5_2', 'pe_15_2', 'psp_1_2', 'psp_3_2']
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='train_method1_2', dict = 'rbc_bak')

# print('begin plot phys scales')
# # # log_list = ['baseline', 'ps_0.01', 'ps_0.05', 'ps_0.2', 'ps_0.5']
# # # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_ps')
# log_list = ['phys_inc', 'ps_0.01', 'ps_0.05', 'ps_0.2']
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='f_ps', dict = 'rbc')

# print('begin plot phys epochs')
# log_list = ['phys_inc', 'pe_5', 'pe_15']
# # test_plot(t_nn, log_list, scale_k, ex_name='fb_0.0', fig_name='f_pe')
# test_plot(t_nn, log_list, scale_k, ex_name='rbc', fig_name='f_pe', dict = 'rbc')

# fig, ax = plt.plot(dpi=1000)

# scale_list = ['1.0', '2.0', '4.0', '8.0']
# log_list = ['data_based', 'phys_inc']
# error1, error2 = [], []
# log_path = f'logs/data_nse/error/phase1_test_data_based_fb_0.0'
# _, error_cul = torch.load(log_path)
# error1.append(error_cul[:, -1].max())
# log_path = f'logs/data_nse/error/phase1_test_phys_inc_fb_0.0'
# _, error_cul = torch.load(log_path)
# error2.append(error_cul[:, -1].max())

# for i in range(1, len(scale_list)):
#     scale = scale_list[i]
#     log_path = f'logs/data_nse/error/phase1_test_data_based_fb_0.0_scale_{scale}'
#     _, error_cul = torch.load(log_path)
#     error1.append(error_cul[:, -1].max())
#     log_path = f'logs/data_nse/error/phase1_test_phys_inc_fb_0.0_scale_{scale}'
#     _, error_cul = torch.load(log_path)
#     error2.append(error_cul[:, -1].max())

# print(error1, error2)

# x = range(len(scale_list))
# rects1 = plt.bar(x = x, height=error2, width=0.4, alpha=0.8,  label='ours')
# rects2 = plt.bar(x = [i+0.4 for i in x], height=error1, width=0.4, label='data-based')
# # plt.ylim(0, 0.3)
# plt.ylabel('error', fontsize=15)
# plt.xticks([i + 0.2 for i in x], scale_list)
# plt.xlabel('random scale of test data', fontsize=15)
# plt.title('error in diffenrent test data', fontsize=20)
# plt.legend()
# plt.savefig('logs/test.jpg')   

# scale_list = ['1.0', '2.0', '4.0']
# log_list = ['data_based', 'phys_inc']
# error1, error2 = [], []
# log_path = f'logs/data_nse/error/phase1_test_data_based_fb_0.0'
# error, _ = torch.load(log_path)
# error1.append(error.mean())
# log_path = f'logs/data_nse/output/phase1_test_data_based_fb_0.0'
# _, loss, _, _ = torch.load(log_path)
# error2.append(torch.sqrt(loss[..., 0] ** 2 + loss[..., 1] ** 2).mean())

# for i in range(1, len(scale_list)):
#     scale = scale_list[i]
#     log_path = f'logs/data_nse/error/phase1_test_data_based_fb_0.0_scale_{scale}'
#     error, _ = torch.load(log_path)
#     error1.append(error.mean())
#     log_path = f'logs/data_nse/output/phase1_test_data_based_fb_0.0_scale_{scale}'
#     _, loss, _, _ = torch.load(log_path)
#     error2.append(torch.sqrt(loss[..., 0] ** 2 + loss[..., 1] ** 2).mean())

# print(error1, error2)

# x = range(len(scale_list))
# rects1 = plt.bar(x = x, height=error1, width=0.4, alpha=0.8,  label='pred loss')
# rects2 = plt.bar(x = [i+0.4 for i in x], height=error2, width=0.4, label='phys loss')
# # plt.ylim(0, 0.3)
# plt.ylabel('error', fontsize=15)
# plt.xticks([i + 0.2 for i in x], scale_list)
# plt.xlabel('random scale of test data', fontsize=15)
# plt.title('error in diffenrent test data', fontsize=20)
# plt.legend()
# plt.savefig('logs/test.jpg')   