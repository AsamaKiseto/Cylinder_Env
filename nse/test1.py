import torch
from scripts.utils import *
from scripts.nse_model import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

tg = 1
Ng = 1
nx = 128
ny = 64
dt = 0.01 * tg
t_start = 0
data_path = 'data/nse_data'
data = ReadData(data_path)
obs, Cd, Cl, ctr = data.split(Ng, tg)
obs_bf = obs[0, t_start:-1].reshape(-1, nx, ny, 3)
obs_af = obs[0, t_start + 1:].reshape(-1, nx, ny, 3)
nk = obs_bf.shape[0]

u_bf = obs_bf[..., :2]
p_bf = obs_bf[..., -1]
u_af = obs_af[..., :2]
p_af = obs_af[..., -1]

u_h = torch.fft.fft2(u_bf, dim=[1, 2])
p_h = torch.fft.fft2(p_bf, dim=[1, 2]).reshape(-1, nx, ny, 1)

k_x = torch.arange(-nx//2, nx//2) * 2 * torch.pi / nx
k_y = torch.arange(-ny//2, ny//2) * 2 * torch.pi / ny
k_x = torch.fft.fftshift(k_x)
k_y = torch.fft.fftshift(k_y)

k_x = k_x.reshape(nx, 1).repeat(1, ny).reshape(1,nx,ny,1)
k_y = k_y.reshape(1, ny).repeat(nx, 1).reshape(1,nx,ny,1)
lap = -(k_x ** 2 + k_y ** 2)

ux_h = 1j * k_x * u_h
uy_h = 1j * k_y * u_h

# print(ux_h.shape) 
px_h = 1j * k_x * p_h
py_h = 1j * k_y * p_h
ulap_h = lap * u_h

ux = torch.fft.ifft2(ux_h, dim=[1, 2])
uy = torch.fft.ifft2(uy_h, dim=[1, 2])
px = torch.fft.ifft2(px_h, dim=[1, 2])
py = torch.fft.ifft2(py_h, dim=[1, 2])
u_lap = torch.fft.ifft2(ulap_h, dim=[1, 2])

ux = torch.real(ux)
uy = torch.real(uy)
px = torch.real(px)
py = torch.real(py)
u_lap = torch.real(u_lap)

p_grad = torch.cat((px, py), -1)
L_state = (u_af - u_bf) / dt + u_bf[..., 0].reshape(-1, nx, ny, 1) * ux + u_bf[..., 1].reshape(-1, nx, ny, 1) * uy - 0.001 * u_lap + p_grad

loss = (L_state ** 2).mean()
L = L_state ** 2
div_u = ux + uy
print(loss)

x = np.linspace(0, 2.2, nx)
y = np.linspace(0, 0.41, ny)
Y, X = np.meshgrid(y, x)
L = L.detach().numpy()
L1 = L[..., 0]
L2 = L[..., 1]
varL = L1 ** 2 + L2 ** 2

fig, ax = plt.subplots(figsize=(22, 4.1), dpi=400)
ax.contourf(X, Y, varL[0])

def animate(i):
    ax.clear()
    ax.contourf(X, Y, varL[i])
    # ax.plot(x[i], y[i])

print('generate anime')
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nk), interval=1, repeat=False)
myAnimation.save('test.gif')

"3D fig"
# fig = plt.figure(figsize=(12,10), dpi=1000)
# ax = plt.axes(projection='3d')
# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2.2, 0.41, 1, 1]))
# ax.plot_surface(X, Y, varL[0], cmap='winter')
# ax.contour(X, Y, varL[0], zdir='z', offset=600, cmap='rainbow')
