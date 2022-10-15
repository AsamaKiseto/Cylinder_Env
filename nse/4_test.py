from scripts.test_utils import *
# tg = args.tg

tg = 5
log_list = ['data_based', 'baseline', 'ps_0.1']

# # loss 
# data_path = 'data/nse_data_reg_dt_0.01_fr_1.0'
# data = LoadData(data_path)
# data.split(1, tg)
# N0, nt, nx, ny = data.get_params()
# loss_log(data, file_name)

# test 
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadData(data_path)
data.split(1, tg)
for file_name in log_list:
    test_log(data, file_name, 'fb_0.0')

data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_2.0'
data = LoadData(data_path)
data.split(1, tg)
for file_name in log_list:
    test_log(data, file_name, 'fb_2.0')

