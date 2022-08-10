import torch
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

data = torch.load('data/nse_data_control_test')
obs, _, _, _ = data
obs = obs.numpy()
print(obs.shape)

x, y, u, v = [obs[:, :, :, i] for i in range(4)]
w = u**2 + v**2
xl, xh  = np.min(x), np.max(x)
yl, yh = np.min(y), np.max(y)

figsizer=10
fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
ax.axis('equal')
# ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
ax.set(xlim=(xl, xh), ylim=(yl, yh))

def animate(i):
    ax.clear()
    ax.quiver(x[i], y[i], u[i], v[i], w[i])
    # ax.plot(x[i], y[i])

myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(400), interval=10, repeat=False)
myAnimation.save('test.gif')