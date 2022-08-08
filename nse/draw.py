import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

data_orig = torch.load('data/baseline_dt_0.01_T_4')
obs_orig, _, _, _, _ = data_orig
obs_orig = obs_orig[3][0].numpy()
print(obs_orig.shape)

x, y, u, v = [obs_orig[:,:,i] for i in range(4)]
w = u**2 + v**2
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)

figsizer=10
fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
ax.axis('equal')
# ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
ax.set(xlim=(xl, xh), ylim=(yl, yh))
ax.quiver(x, y, u, v, w)
print('end')
