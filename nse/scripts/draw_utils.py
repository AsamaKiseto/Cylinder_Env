from ast import Str
from scripts.utils import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch

# draw loss plot
def loss_plot(log_list, fig_name = 'test'):
    # fig setting
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(15,12), dpi=500)
    ax = ax.flatten()
    # plt.figure(figsize=(15, 12))
    for i in range(3):
        ax[i] = plt.subplot2grid((3, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        ax[i].set_yscale('log')
        # ax[i].set_ylabel(f'loss{i+1}', fontsize=20)
        ax[i].set_ylim(1e-4, 1)

    ax[0].set_title("loss plot", fontsize=20)
    ax[0].set_ylabel("pred loss", fontsize=20)
    ax[1].set_ylabel("phys loss of obs", fontsize=20)
    ax[2].set_ylabel("phys loss of pred", fontsize=20)
    ax[2].set_xlabel("epochs", fontsize=20)

    for k in range(len(log_list)):
        loss = torch.load('logs/data/loss_log_' + log_list[k])
        for i in range(3):
            ax[i].plot(loss[i], label=log_list[k])
            ax[i].legend()
    
    plt.savefig(f'logs/loss_plot_{fig_name}.jpg')


def test_plot(t_nn, log_list, scale_k, ex_name = 'fb_0.0', fig_name = 'test', dict = 'nse', label_list = None):
    if label_list==None:
        label_list = log_list
        
    # state error fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=1000)
    ax = ax.flatten()
    
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e-3, 1)
    
    # ax[1].set_ylim(0, 0.1)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel("One-step data error", fontsize=20)
    ax[1].set_ylabel("Cumul data error", fontsize=20)
    ax[fig_num - 1].set_xlabel("t", fontsize=20)

    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data_{dict}/error/phase1_test_{log_list[k]}_{ex_name}')

        error_1step, error_cul = calMean(data_list)
        error_1step_v, error_cul_v = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_1step[scale_k[j]], label = f'{label_list[k]}')
            ax[1].plot(t_nn, error_cul[scale_k[j]], label = f'{label_list[k]}')

            ax[0].fill_between(t_nn, error_1step_v[0][scale_k[j]], error_1step_v[1][scale_k[j]], alpha=0.2)
            ax[1].fill_between(t_nn, error_cul_v[0][scale_k[j]], error_cul_v[1][scale_k[j]], alpha=0.2)
            
            ax[0].legend()
    
    plt.savefig(f'logs/pics_{dict}/error/phase1_state_{fig_name}_{ex_name}.jpg')

    # # coef fig setting
    # fig_num = 2
    # fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    # ax = ax.flatten()

    # for i in range(fig_num):
    #     ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
    #     ax[i].grid(True, lw=0.4, ls="--", c=".50")
    #     # ax[i].set_xlim(0, t_nn[-1])
    #     ax[i].set_yscale('log')
    #     ax[i].set_ylim(1e-4, 1)
        
    # ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    # ax[0].set_ylabel(r'Cul $C_D$ error', fontsize=20)
    # ax[1].set_ylabel(r'Cul $C_L$ error', fontsize=20)
    # ax[fig_num-1].set_xlabel("t", fontsize=20)
    
    # for k in range(len(log_list)):
    #     if bak:
    #         data_list = torch.load(f'logs/data_bak/error/phase1_test_{log_list[k]}_{ex_name}')
    #     else:
    #         data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
    #     _, _, _, _, error_Cd_cul, error_Cl_cul = calMean(data_list)
    #     _, _, _, _, error_Cd_cul_v, error_Cl_cul_v = calVar(data_list)
        
    #     for j in range(len(scale_k)):
    #         ax[0].plot(t_nn, error_Cd_cul[scale_k[j]], label = f'{log_list[k]}')
    #         ax[1].plot(t_nn, error_Cl_cul[scale_k[j]], label = f'{log_list[k]}')
            
    #         ax[0].fill_between(t_nn, error_Cd_cul_v[0][scale_k[j]], error_Cd_cul_v[1][scale_k[j]], alpha=0.2)
    #         ax[1].fill_between(t_nn, error_Cl_cul_v[0][scale_k[j]], error_Cl_cul_v[1][scale_k[j]], alpha=0.2)
            
    #         ax[0].legend()
            
    # if bak:
    #     plt.savefig(f'logs/pics_bak/error/phase1_culcoef_{fig_name}_{ex_name}.jpg')
    # else:
    #     plt.savefig(f'logs/pics/error/phase1_culcoef_{fig_name}_{ex_name}.jpg')


def test_plot1(t_nn, log_list, scale_k, ex_name = 'fb_0.0', fig_name = 'test', dict = 'nse', zlim=1, label_list = None):
    # state error fig setting
    fig_num = 1
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=1000)
    
    if label_list==None:
        label_list = log_list

    ax = plt.subplot2grid((fig_num, 1), (0, 0))
    ax.grid(True, lw=0.4, ls="--", c=".50")
    # ax[i].set_xlim(0, t_nn[-1])
    # ax.set_yscale('log')
    
    ax.set_ylim((0, zlim))
        
    ax.set_title("Cumulative state error", fontsize=40)
    ax.set_ylabel("relative error", fontsize=40)
    ax.set_xlabel("t", fontsize=40)

    for k in range(len(log_list)):
        data_list = torch.load(f'logs/data_{dict}/error/phase1_test_{log_list[k]}_{ex_name}')

        error_1step, error_cul = calMean(data_list)
        error_1step_v, error_cul_v = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax.plot(t_nn, error_cul[scale_k[j]], label = f'{label_list[k]}')
            ax.fill_between(t_nn, error_cul_v[0][scale_k[j]], error_cul_v[1][scale_k[j]], alpha=0.2)
            ax.legend(fontsize=20)
    
    plt.savefig(f'logs/pics_{dict}/error/phase1_state_{fig_name}_{ex_name}.jpg')


def test_plot_ts(t_nn, log_list, scale_k, ts_list, ex_name = 'fb_0.0', fig_name = 'test'):
    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_yscale('log')
        ax[i].set_ylim(0, 0.5)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel(r'Cul $C_D$ error', fontsize=20)
    ax[1].set_ylabel(r'Cul $C_L$ error', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)
    
    data_list = torch.load(f'logs/data/phase1_test_{log_list[0]}_{ex_name}')
    _, _, error_Cd_cul, error_Cl_cul = calMean(data_list)
    _, _, error_Cd_cul_v, error_Cl_cul_v = calVar(data_list)
    
    for j in range(len(scale_k)):
        ax[0].plot(t_nn, error_Cd_cul[scale_k[j]], label = f'{log_list[0]}')
        ax[1].plot(t_nn, error_Cl_cul[scale_k[j]], label = f'{log_list[0]}')
        
        ax[0].fill_between(t_nn, error_Cd_cul_v[0][scale_k[j]], error_Cd_cul_v[1][scale_k[j]], alpha=0.2)
        ax[1].fill_between(t_nn, error_Cl_cul_v[0][scale_k[j]], error_Cl_cul_v[1][scale_k[j]], alpha=0.2)
        
        ax[0].legend()

    for k in range(1, len(log_list)):
        for ts in ts_list:
            # data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
            data_list = torch.load(f'logs/data/phase1_test_ts_{ts}_{log_list[k]}_{ex_name}')
            _, _, error_Cd_cul, error_Cl_cul = calMean(data_list)
            _, _, error_Cd_cul_v, error_Cl_cul_v = calVar(data_list)
            
            for j in range(len(scale_k)):
                ax[0].plot(t_nn, error_Cd_cul[scale_k[j]], label = f'{log_list[k]}, ts={ts}')
                ax[1].plot(t_nn, error_Cl_cul[scale_k[j]], label = f'{log_list[k]}, ts={ts}')
                
                ax[0].fill_between(t_nn, error_Cd_cul_v[0][scale_k[j]], error_Cd_cul_v[1][scale_k[j]], alpha=0.2)
                ax[1].fill_between(t_nn, error_Cl_cul_v[0][scale_k[j]], error_Cl_cul_v[1][scale_k[j]], alpha=0.2)
                
                ax[0].legend()
            
    plt.savefig(f'logs/pics/error/phase1_coef_cul_{fig_name}_{ex_name}.jpg')    
    
def coef_plot(t_nn, scale_k, data, fig_name):
    scale = [0.1, 0.5, 1.0]
    
    Cd_mean, Cl_mean = calMean(data)
    Cd_var, Cl_var = calVar(data)

    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_yscale('log')
        # ax[i].set_ylim(5e-4, 5)
        
    ax[0].set_title("Test Obs", fontsize=20)
    ax[0].set_ylabel(r'$C_D$', fontsize=20)
    ax[1].set_ylabel(r'$C_L$', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)

    for j in range(len(scale_k)):
        ax[0].plot(t_nn, Cd_mean[scale_k[j]], label=scale[scale_k[j]])
        ax[1].plot(t_nn, Cl_mean[scale_k[j]], label=scale[scale_k[j]])
        
        ax[0].fill_between(t_nn, Cd_var[0][scale_k[j]], Cd_var[1][scale_k[j]], alpha=0.2)
        ax[1].fill_between(t_nn, Cl_var[0][scale_k[j]], Cl_var[1][scale_k[j]], alpha=0.2)

    plt.savefig(f'logs/pics/obs_coef_{fig_name}.jpg')
    
def coef_plot1(t_nn, data, fig_name):
    scale = [0.1, 0.5, 1.0]
    
    Cd, Cl = data

    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        # ax[i].set_yscale('log')
        # ax[i].set_ylim(5e-4, 5)
        
    ax[0].set_title("Test Obs", fontsize=20)
    ax[0].set_ylabel(r'$C_D$', fontsize=20)
    ax[1].set_ylabel(r'$C_L$', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)

    for i in range(Cd.shape[0]):
        ax[0].plot(t_nn, Cd[i])
        ax[1].plot(t_nn, Cl[i])

    plt.savefig(f'logs/pics/obs_coef_{fig_name}_all.jpg')
    
def animate_field(data, xy_mesh, name, file_name, dict='nse'):
    nt = data.shape[0]
    
    x, y, xl, xh, yl, yh = xy_mesh

    figsizer=20
    fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
    ax.axis('equal')
    ax.set(xlim=(xl, xh), ylim=(yl, yh))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    u, v = [data[:, :, :, i] for i in range(2)]
    w = torch.sqrt(u**2 + v**2)

    def animate(i):
        ax.clear()
        ax.set_title(f'{name} {file_name}')
        ax.quiver(x, y, u[i], v[i], w[i])
        # ax.contourf(x, y, w[i])
        # ax.plot_surface(x, y, Lpde_obs[i, :, :, 0])
        # ax.plot(x[i], y[i])
        
    print(f'generate anime {name}')
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_{dict}/output/{file_name}_{name}.gif')
    
    
def animate2D(data, xy_mesh, name, file_name, dict='nse'):
    nt = data.shape[0]
    print(data.shape)
    
    x, y, xl, xh, yl, yh = xy_mesh

    figsizer=10
    fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
    ax.axis('equal')
    # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
    ax.set(xlim=(xl, xh), ylim=(yl, yh))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    def animate(i):
        ax.clear()
        ax.set_title(f'{name} {file_name}')
        # print(data[i].shape)
        # ax.quiver(x, y, u[i], v[i], w[i])
        ax.contourf(x, y, data[i], 200, cmap='jet')
        # plt.colorbar()
        # ax.plot_surface(x, y, Lpde_obs[i, :, :, 0])
        # ax.plot(x[i], y[i])
        
    print(f'generate anime {name}')
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_{dict}/output/{file_name}_{name}.gif')

def animate3D(data, xy_mesh, name, file_name, zlim = 100, dict = 'nse'):
    nt = data.shape[0]
    
    x, y, xl, xh, yl, yh = xy_mesh

    figsizer=10
    # fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
    # ax.axis('equal')
    # # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
    # ax.set(xlim=(xl, xh), ylim=(yl, yh))

    fig = plt.figure(figsize = (16, 10))
    ax = plt.axes(projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([(xh - xl)*figsizer, (yh-yl)*figsizer, figsizer*1.2, figsizer*2]))
    
    u, v = [data[:, :, :, i] for i in range(2)]
    w = torch.sqrt(u**2 + v**2)

    def animate(i):
        ax.clear()
        # ax.set_title(f'{name} {file_name}')
        # ax.quiver(x, y, u[i], v[i], w[i])
        # ax.set_zlim(-zlim, zlim)
        ax.set_zlim(0, zlim)
        ax.plot_surface(x, y, w[i], cmap='rainbow')
        # ax.plot(x[i], y[i])
        
    print(f'generate anime {name}')
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_{dict}/output/{file_name}_{name}.gif')

def animate2D_comp(obs, log_list, num_k, xy_mesh, name='comp1', ex_name = 'fb_0.0', dict='nse'):
    x, y, xl, xh, yl, yh = xy_mesh
    nt, nx, ny = obs.shape[1], obs.shape[2], obs.shape[3]
    u = obs[..., 0]
    v = obs[..., 1]
    uv = u ** 2 + v ** 2
    p = obs[..., 2]
    fontsize = 25
    
    nump = len(log_list)
    u_, v_, p_, uv_ = torch.zeros(nump, 30, nt, nx, ny), torch.zeros(nump, 30, nt, nx, ny), torch.zeros(nump, 30, nt, nx, ny), torch.zeros(nump, 30, nt, nx, ny)
    for i in range(nump):
        out_cul, _, _, _ = torch.load(f'logs/data_{dict}/output/phase1_test_{log_list[i]}_{ex_name}')
        u_[i] = out_cul[..., 0]
        v_[i] = out_cul[..., 1]
        uv_[i] = u_[i] ** 2 + v_[i] ** 2
        p_[i] = out_cul[..., 2]

    figsizer=10
    fig, ax = plt.subplots(nrows=nump+1, ncols=1, figsize=((xh - xl)*figsizer,(yh-yl)*figsizer*(nump+2)))
    ax = ax.flatten()
    for i in range(3):
        ax[i].axis('equal')
        # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
        ax[i].set(xlim=(xl, xh), ylim=(yl, yh))

    def animate(k):
        ax[0].clear()
        ax[0].set_title('u', fontsize=fontsize*1.5)
        ax[0].set_ylabel('obs', fontsize=fontsize)
        ax[0].contourf(x, y, u[num_k, k], 200, cmap='jet')
        for i in range(nump):
            ax[i+1].clear()
            ax[i+1].set_ylabel(f'{log_list[i]}', fontsize=fontsize)
            ax[i+1].contourf(x, y, u_[i, num_k, k], 200, cmap='jet')

    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_{dict}/output/u_{name}.gif')

    figsizer=10
    fig, ax = plt.subplots(nrows=nump+1, ncols=1, figsize=((xh - xl)*figsizer,(yh-yl)*figsizer*(nump+2)))
    ax = ax.flatten()
    for i in range(3):
        ax[i].axis('equal')
        # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
        ax[i].set(xlim=(xl, xh), ylim=(yl, yh))

    def animate(k):
        ax[0].clear()
        ax[0].set_title('v', fontsize=fontsize*1.5)
        ax[0].set_ylabel('obs', fontsize=fontsize)
        ax[0].contourf(x, y, v[num_k, k], 200, cmap='jet')
        for i in range(nump):
            ax[i+1].clear()
            ax[i+1].set_ylabel(f'{log_list[i]}', fontsize=fontsize)
            ax[i+1].contourf(x, y, v_[i, num_k, k], 200, cmap='jet')

    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_{dict}/output/v_{name}.gif')

    figsizer=10
    fig, ax = plt.subplots(nrows=nump+1, ncols=1, figsize=((xh - xl)*figsizer,(yh-yl)*figsizer*(nump+2)))
    ax = ax.flatten()
    for i in range(3):
        ax[i].axis('equal')
        # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
        ax[i].set(xlim=(xl, xh), ylim=(yl, yh))

    def animate(k):
        ax[0].clear()
        ax[0].set_title('p', fontsize=fontsize*1.5)
        ax[0].set_ylabel('obs', fontsize=fontsize)
        ax[0].contourf(x, y, p[num_k, k], 200, cmap='jet')
        for i in range(nump):
            ax[i+1].clear()
            ax[i+1].set_ylabel(f'{log_list[i]}', fontsize=fontsize)
            ax[i+1].contourf(x, y, p_[i, num_k, k], 200, cmap='jet')

    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_{dict}/output/p_{name}.gif')