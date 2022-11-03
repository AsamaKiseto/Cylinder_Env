from scripts.test_utils import *

scale = [0.1, 0.5, 1.0]
tg = 5

# test 
# log_list = ['data_based', 'phys_inc', 'no_random', 'random_select_0.001']
# data_path = 'data/test_data/nse_data_reg_dt_0.01_fb_0.0'
# data = LoadDataNSE(data_path)
# data.split(1, tg)
# for file_name in log_list:
#     test_log(data, file_name, 'fb_0.0', model_loaded = NSEModel_FNO, dict='nse')
# test_log(data, 'phys_bak', 'fb_0.0', model_loaded = NSEModel_FNO_prev, dict='nse')

log_list = ['data_based', 'phys_inc']
scale_list = ['2.0', '4.0', '6.0', '8.0', '10.0']


for scale in scale_list:
    for file_name in log_list:
        data_path = f'data/test_data/nse_data_reg_dt_0.01_fb_0.0_scale_{scale}'
        data = LoadDataNSE(data_path)
        data.split(1, tg)
        test_log(data, file_name, f'fb_0.0_scale_{scale}', model_loaded = NSEModel_FNO, dict='nse')


log_list = ['data_based', 'phys_inc', 'no_random', 'random_select_0.001']
# log_list = ['pe_5', 'pe_15', 'ps_0.01', 'ps_0.1']
data_path = 'data/test_data/nse_data_reg_rbc'
data_path = 'data/nse_data_reg_rbc'
data = LoadDataRBC(data_path)
# data.split()

for file_name in log_list:
    file_name = file_name
    test_log(data, file_name,  'rbc', model_loaded = RBCModel_FNO,  dict = 'rbc', dt = 0.01)
# test_log(data, 'prev_phys', 'rbc', model_loaded = RBCModel_FNO_prev,  dict = 'rbc', dt = 0.01)
