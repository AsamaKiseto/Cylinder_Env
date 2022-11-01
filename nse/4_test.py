from scripts.test_utils import *

scale = [0.1, 0.5, 1.0]
tg = 5

# test 
log_list = ['phys_inc', 'no_random', 'random_select_0.001']
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadDataNSE(data_path)
data.split(1, tg)
for file_name in log_list:
    test_log(data, file_name, 'fb_0.0', model_loaded = NSEModel_FNO, dict='nse')

# test_log(data, 'prev_phys', 'fb_0.0', model_loaded = NSEModel_FNO_prev, dict='nse')

# log_list = ['data_based', 'phys_inc', 'pe_5', 'no_random', 'random_select_0.001', 'random_select_0.0001']
# data_path = 'data/test_data/nse_data_reg_rbc'
# # data_path = 'data/nse_data_reg_rbc'

# data = LoadDataRBC(data_path)
# # data.split()

# for file_name in log_list:
#     file_name = file_name
#     test_log(data, file_name,  'rbc', model_loaded = RBCModel_FNO,  dict = 'rbc')
# test_log(data, 'prev_phys', 'rbc', model_loaded = RBCModel_FNO_prev,  dict = 'rbc')
