import torch

data = torch.load('data/nse_data_N0_256_nT_400')
obs, reward, Cd, Cl, f = data
print(obs.shape)
print(f[0])

obs_init = obs[0, 0]
for i in range(256):
    obs[i, 0, :, :] = obs_init

print(obs.shape)
data = [obs, reward, Cd, Cl, f]

torch.save(data, 'data/nse_data')