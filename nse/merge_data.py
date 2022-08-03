import torch

data1_path = 'nse_data_N0_64_nT_400_f1_-2.0_f2_-2.0'
data2_path = 'nse_data_N0_64_nT_400_f1_0.0_f2_-2.0'
data3_path = 'nse_data_N0_64_nT_400_f1_-2.0_f2_0.0'
data4_path = 'nse_data_N0_64_nT_400_f1_0.0_f2_0.0'

def split_data(data_path):
    data = torch.load('data/'+data_path)
    obs, r, Cd, Cl, f = data
    f = torch.Tensor(f)
    return obs, r, Cd, Cl, f

def merge(a, a2m):
    return torch.cat((a, a2m), dim=0)

obs, r, Cd, Cl, f = split_data(data1_path)
print(obs.shape)
for i in range(2, 5):
    exec(f'obs{i}, r{i}, Cd{i}, Cl{i}, f{i} = split_data(data{i}_path)')
    exec(f'obs = merge(obs, obs{i})')
    exec(f'r = merge(r, r{i})')
    exec(f'Cd = merge(Cd, Cd{i})')
    exec(f'Cl = merge(Cl, Cl{i})')
    exec(f'f = merge(f, f{i})')

data = [obs, r, Cd, Cl, f]
print(obs.shape)
torch.save(data, 'data/nse_data_N0_256_nT_400')