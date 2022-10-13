import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

data = torch.load('data/nse_data_reg_dt_0.01_fr_1.0')

log_list = ['data_based', 'ps_0.1']
# data_list = torch.load(f'logs/data/phase1_test_{log_list[k]}_{ex_name}')
data_list = torch.load(f'logs/data/phase1_test_data_based_fb_0.0')
error_1step, Lpde_obs, Lpde_pred, error_cul, Lpde_pred_cul, error_Cd_1step, error_Cl_1step, error_Cd_cul, error_Cl_cul = data_list

# obs, _, _, _, _ = data
# obs = obs.numpy()
# print(obs.shape)
# nT = obs.shape[0]

# x, y, u, v = [obs[:, :, :, i] for i in range(4)]
# w = u**2 + v**2
# xl, xh  = np.min(x), np.max(x)
# yl, yh = np.min(y), np.max(y)

# figsizer=10
# fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
# ax.axis('equal')
# # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
# ax.set(xlim=(xl, xh), ylim=(yl, yh))

# def animate(i):
#     ax.clear()
#     ax.quiver(x[i], y[i], u[i], v[i], w[i])
#     # ax.plot(x[i], y[i])

# print('generate anime')
# myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nT), interval=1, repeat=False)
# myAnimation.save('test.gif')

x = np.arange(256) / 256 * 2.2
y = np.arange(64) / 64 * 0.41
x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = plt.axes(projection='3d')

def animate(i):
    ax.clear()
    ax.plot_surface(x, y, )
    # ax.plot(x[i], y[i])