from scripts.test_utils import *
# tg = args.tg

tg = 5
# log_list = ['data_based', 'baseline', 'ps_0.1']
# log_list = ['ps_0.2', 'ps_0.3']
log_list = ['pe_20', 'pe_30', 'pe_40', 'pe_50', 'ps_0.01', 'ps_0.03', 'ps_0.08']

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

