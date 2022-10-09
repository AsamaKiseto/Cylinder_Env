import torch

scale = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
name = ["nse_data_reg_dt_0.01_fb_0.0", "nse_data_reg_dt_0.01_fb_1.0", "nse_data_reg_dt_0.02_fb_0.0", "nse_data_reg_dt_0.02_fb_1.0"]

for j in range(len(name)):
    for i in range(10):
        exec(f'data{i}_path = name[{j}] + "_scale_{scale[i]}"')
        exec(f'print(data{i}_path)')

    def split_data(data_path):
        obs, Cd, Cl, ctr = torch.load('data/test_data/'+data_path)
        return obs, Cd, Cl, ctr

    def merge(a, a2m):
        return torch.cat((a, a2m), dim=0)

    obs, Cd, Cl, ctr = split_data(data0_path)
    print(obs.shape)
    for i in range(1, 10):
        exec(f'obs{i},  Cd{i}, Cl{i}, ctr{i} = split_data(data{i}_path)')
        exec(f'obs = merge(obs, obs{i})')
        exec(f'Cd = merge(Cd, Cd{i})')
        exec(f'Cl = merge(Cl, Cl{i})')
        exec(f'ctr = merge(ctr, ctr{i})')

    data = [obs, Cd, Cl, ctr]
    print(obs.shape)
    torch.save(data, 'data/test_data/' + name[j])