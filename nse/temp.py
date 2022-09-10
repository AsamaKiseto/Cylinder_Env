import torch

obs, reward, Cd, Cl, f = torch.load('data/nse_data')
# obs2, Cd2, Cl2, f2 = torch.load('data/nse_data_sparse')
# print(obs.shape)

# obs1 = obs[125:132]
# Cd1 = Cd[125:132]
# Cl1 = Cl[125:132]
# f1 = f[125:132]

# print(f1.shape)

# obs = torch.cat((obs2, obs1))
# print(obs.shape)

# Cd = torch.cat((Cd2, Cd1))
# Cl = torch.cat((Cl2, Cl1))
# f = torch.cat((f2, f1))

# print(Cd.shape, Cl.shape, f.shape)
data = [obs, Cd, Cl, f]

torch.save(data, 'data/nse_data')


import torch
op = 'logs/phase1_ex0_grid_pi'
state_dict, logs = torch.load(op)
logs.keys()
logs['modify'] = True
torch.save([state_dict, logs], op)