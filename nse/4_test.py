from scripts.test_utils import *

scale = [0.1, 0.5, 1.0]
tg = 5
# log_list = ['data_based', 'baseline', 'ps_0.1']
# log_list = ['ps_0.2', 'ps_0.3']

# # loss 
# log_list = ['no_random', 'random_select', 'dr_0.3', 'dr_0.5']
# data_path = 'data/nse_data_reg_dt_0.01_fr_1.0'
# data = LoadData(data_path)
# data.split(1, tg)
# N0, nt, nx, ny = data.get_params()
# for file_name in log_list:
#     loss_log(data, file_name)

# test 
log_list = ['dr_0.3_0', 'dr_0.5_0']
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadData(data_path)
data.split(1, tg)
for file_name in log_list:
    test_log1(data, file_name, 'fb_0.0')

# log_list = ['data_based', 'baseline', 'pe_5', 'pe_15', 'pe_20', 'ps_0.01', 'ps_0.03', 'ps_0.05', 'ps_0.2', 'ps_0.3', 'no_random', 'random_select', 'dr_0.3', 'dr_0.5']
log_list = ['data_based', 'baseline', 'data_based_dr_0.3', 'baseline_dr_0.3', 'data_based_dr_0.5', 'baseline_dr_0.5', \
            'pe_5', 'pe_15', 'pe_20', 'ps_0.01', 'ps_0.05', 'ps_0.2', 'ps_0.5', \
            'no_random', 'random_select_0.01', 'random_select_0.001', 'random_select_0.0001', 'psp_1', 'psp_3']
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadData(data_path)
data.split(1, tg)
for file_name in log_list:
    test_log(data, file_name, 'fb_0.0')
