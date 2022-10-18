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
log_list = ['pe_5', 'no_random', 'random_select', 'dr_0.3', 'dr_0.5']
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadData(data_path)
data.split(1, tg)
for file_name in log_list:
    test_log(data, file_name, 'fb_0.0')

# data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_2.0'
# data = LoadData(data_path)
# data.split(1, tg)
# for file_name in log_list:
#     test_log(data, file_name, 'fb_2.0')

