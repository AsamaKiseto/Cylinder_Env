import torch

scale_list = [0.1, 1.0, 10.0]
name_list = ["nse_data_reg_dt_0.01_fb_0.0"]

for name in name_list:
    data_path = []
    for scale in scale_list:
        data_path.append(f'{name}_scale_{scale}')
        print(data_path[-1])

    def split_data(data_path):
        obs, Cd, Cl, ctr = torch.load(data_path)
        return obs, Cd, Cl, ctr

    def merge(a, a2m):
        return torch.cat((a, a2m), dim=0)

    obs, Cd, Cl, ctr = split_data(data_path[0])
    print(obs.shape)
    for i in range(1, len(data_path)):
        obs0,  Cd0, Cl0, ctr0 = split_data(data_path[i])
        obs = merge(obs, obs0)
        Cd = merge(Cd, Cd0)
        Cl = merge(Cl, Cl0)
        ctr = merge(ctr, ctr0)

    data = [obs, Cd, Cl, ctr]
    print(obs.shape)
    torch.save(data, f'{name}_1')
