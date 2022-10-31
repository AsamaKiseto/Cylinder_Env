from scripts.test_utils import *

scale = [0.1, 0.5, 1.0]
tg = 5

# test 
# log_list = ['dr_0.3_0', 'dr_0.5_0']
data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
data = LoadDataNSE(data_path)
data.split(1, tg)
# for file_name in log_list:
#     test_log(data, file_name, 'fb_0.0', dict='nse')

test_log(data, 'prev_phys', 'fb_0.0', model_loaded = NSEModel_FNO_prev, dict='nse')

# # log_list = ['data_based', 'baseline', 'pe_5', 'pe_15', 'pe_20', 'ps_0.01', 'ps_0.03', 'ps_0.05', 'ps_0.2', 'ps_0.3', 'no_random', 'random_select', 'dr_0.3', 'dr_0.5']
# log_list = ['data_based', 'phys_inc', 'pe_5', 'pe_15', 'ps_0.01', 'ps_0.05', 'ps_0.2', \
#             'no_random', 'random_select_0.001', 'random_select_0.0001']
# data_path = 'data/test_data/nse_data_reg_rbc'
# # data_path = 'data/nse_data_reg_rbc'

# data = LoadDataRBC(data_path)
# # data.split()

# for file_name in log_list:
#     file_name = 'rbc_' + file_name
#     test_log(data, file_name,  'rbc', model_loaded = RBCModel_FNO,  dict = 'rbc')
# test_log(data, 'rbc_prev_phys', 'rbc', model_loaded = RBCModel_FNO_prev,  dict = 'rbc')
