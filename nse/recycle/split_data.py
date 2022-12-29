import torch
import os
from timeit import default_timer

scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
name = ["nse_data_reg_dt_0.01_fb_0.0", "nse_data_reg_dt_0.01_fb_1.0", "nse_data_reg_dt_0.02_fb_0.0", "nse_data_reg_dt_0.02_fb_1.0"]

for j in range(len(name)):
    print(name[j])

    for i in range(10):
        t1 = default_timer()
        obs, Cd, Cl, ctr= torch.load(name[j])
        data = [obs[10 * i: 10 *(i + 1)], Cd[10 * i: 10 *(i + 1)], Cl[10 * i: 10 *(i + 1)], ctr[10 * i: 10 *(i + 1)]]
        log_file = name[j] + "_scale_" + str(scale[i])
        if not os.path.isfile(log_file):
            torch.save(data, log_file)
        t2 = default_timer()
        print(f'# {i} finished | {t2 - t1}')

