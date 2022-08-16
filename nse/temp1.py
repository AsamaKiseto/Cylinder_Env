import torch
import matplotlib.pyplot as plt 
import numpy as np

# plot colors
from matplotlib import colors

cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)

obs, _, Cd, Cl, _ = torch.load('data/nse_data')
print('data load completed')

plt.figure(figsize=(12,10))
ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

Nk = 32
k = np.arange(Nk) * (256//Nk)
t = np.arange(400) * 0.01

for i in range(Nk):

    ax1.plot(t, Cd[i], color = cmap(i/(Nk+1)))
    ax1.grid(True, lw=0.4, ls="--", c=".50")
    ax1.set_ylabel(r"$Cd$", fontsize=15)
    ax1.set_xlim(0, 4)

    ax2.plot(t, Cl[i], color = cmap(i/(Nk+1)))
    ax2.grid(True, lw=0.4, ls="--", c=".50")
    ax2.set_ylabel(r"$Cl$", fontsize=15)
    ax2.set_xlim(0, 4)

plt.savefig(f'coef_data_test.jpg')