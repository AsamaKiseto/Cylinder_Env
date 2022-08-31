import torch
from scripts.utils import *
from scripts.nse_model import *
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

tg = 1
nx = 41
ny = 41
dt = 0.01 * tg
t_start = 0
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
Y, X = np.meshgrid(y, x)

data_path = 'cavity_data.npy'
obs = np.load(data_path)    # 100, 41, 41, 3
obs = torch.Tensor(obs)

obs_bf = obs[t_start:-1].reshape(-1, nx, ny, 3)
obs_af = obs[t_start + 1:].reshape(-1, nx, ny, 3)
nk = obs_bf.shape[0]

u_bf = obs_bf[..., :2]
p_bf = obs_bf[..., -1]
u_af = obs_af[..., :2]
p_af = obs_af[..., -1]

u_h = torch.fft.fft2(u_bf, dim=[1, 2])
p_h = torch.fft.fft2(p_bf, dim=[1, 2]).reshape(-1, nx, ny, 1)

k_x = torch.arange(-nx//2, nx//2) * 2 * torch.pi
k_y = torch.arange(-ny//2, ny//2) * 2 * torch.pi
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

ux = torch.real(torch.fft.ifft2(ux_h, dim=[1, 2]))
uy = torch.real(torch.fft.ifft2(uy_h, dim=[1, 2]))
px = torch.real(torch.fft.ifft2(px_h, dim=[1, 2]))
py = torch.real(torch.fft.ifft2(py_h, dim=[1, 2]))
u_lap = torch.real(torch.fft.ifft2(ulap_h, dim=[1, 2]))

# ux[:, mask] = 0.0
# uy[:, mask] = 0.0
# px[:, mask] = 0.0
# py[:, mask] = 0.0
# u_lap[:, mask] = 0.0

p_grad = torch.cat((px, py), -1)
L_state = (u_af - u_bf) / dt + u_bf[..., 0].reshape(-1, nx, ny, 1) * ux + u_bf[..., 1].reshape(-1, nx, ny, 1) * uy - 0.001 * u_lap + p_grad

loss = (L_state ** 2).mean()
L = L_state ** 2
div_u = ux + uy
vardiv = np.sqrt((div_u ** 2).sum(-1))
vard = vardiv.sum(0)
print(loss)

# FDM
duFDx = torch.zeros(u_bf.shape)
for i in range(nx-1):
    duFDx[:, i] = (u_bf[:, i+1] - u_bf[:, i]) / 2.2 * nx
duFDx[:, -1] = duFDx[:, -2]
duFDy = torch.zeros(u_bf.shape)
for i in range(ny-1):
    duFDy[:, :, i] = (u_bf[:, :, i+1] - u_bf[:, :, i]) / 0.41 * ny
duFDy[:, :, -1] = duFDy[:, :, -2]

div_uFD = duFDx + duFDy
vardivFD = np.sqrt((div_uFD ** 2).sum(-1))

dduFDx = torch.zeros(u_bf.shape)
for i in range(nx-1):
    dduFDx[:, i] = (duFDx[:, i+1] - duFDx[:, i]) / 2.2 * nx
dduFDx[:, -1] = dduFDx[:, -2]
dduFDy = torch.zeros(u_bf.shape)
for i in range(ny-1):
    dduFDy[:, :, i] = (duFDy[:, :, i+1] - duFDy[:, :, i]) / 0.41 * ny
dduFDy[:, :, -1] = dduFDy[:, :, -2]

dpFDx = torch.zeros(p_bf.shape)
for i in range(nx-1):
    dpFDx[:, i] = (p_bf[:, i+1] - p_bf[:, i]) / 2.2 * nx
dpFDx[:, -1] = dpFDx[:, -2]
dpFDy = torch.zeros(p_bf.shape)
for i in range(ny-1):
    dpFDy[:, :, i] = (p_bf[:, :, i+1] - p_bf[:, :, i]) / 0.41 * ny
dpFDy[:, :, -1] = dpFDy[:, :, -2]

u_lapFD = dduFDx + dduFDy
p_gradFD = torch.cat((dpFDx.reshape(-1, nx, ny, 1), dpFDy.reshape(-1, nx, ny, 1)), -1)

L_FDM = (u_af - u_bf) / dt + u_bf[..., 0].reshape(-1, nx, ny, 1) * duFDx + u_bf[..., 1].reshape(-1, nx, ny, 1) * duFDy - 0.001 * u_lapFD + p_gradFD
varLFD = (L_FDM ** 2).sum(-1)
varLFD = varLFD.detach().numpy()
varLFD = np.sqrt(varLFD)


varux = ux - duFDx
varuy = uy - duFDy
varuxy = np.sqrt((varux**2).sum(-1) + (varuy**2).sum(-1))


L = L.detach().numpy()
varL = L.sum(-1)
varL = np.sqrt(varL)

fig, ax = plt.subplots(figsize=(22, 4.1))
ax.contourf(X, Y, varL[0])

def animate(i):
    ax.clear()
    ax.plot_surface(X, Y, varL[i], cmap='rainbow')

print('generate anime')
myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nk), interval=1, repeat=False)
myAnimation.save('test.gif')

"3D fig"
fig = plt.figure(dpi=400)
ax = plt.axes(projection='3d')
ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2.2, 0.41, 1, 2.2]))
ax.plot_surface(X, Y, varL[0], cmap='rainbow')
ax.contour(X, Y, varL[0], zdir='z', offset=600, cmap='rainbow')

def gen_anime(Z, zmin=0, zmax=100):
    fig = plt.figure(dpi=400)
    ax = plt.axes(projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2.2, 0.41, 1, 2.2]))
    
    def animate(i):
        ax.clear()
        ax.set_zlim(zmin, zmax)
        ax.plot_surface(X, Y, Z[i], cmap='rainbow')
        # ax.plot(x[i], y[i])
    anime = animation.FuncAnimation(fig, animate, frames=np.arange(nk), interval=1, repeat=False)
    
    return anime

varuxy_gif = gen_anime(varuxy)
varuxy_gif.save('varuxy.gif')
varL_gif = gen_anime(varL)
varL_gif.save('varL.gif')

# fig = plt.figure(figsize=(12,10), dpi=1000)
# ax = plt.axes(projection='3d')
# ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2.2, 0.41, 1, 1]))

# def animate(i):
#     ax.clear()
#     ax.plot_surface(X, Y, varL[i], cmap='winter')
