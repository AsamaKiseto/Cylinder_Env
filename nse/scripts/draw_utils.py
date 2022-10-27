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


def test_plot(t_nn, log_list, scale_k, ex_name = 'fb_0.0', fig_name = 'test', bak = False):
    if bak:
        scale = [0.1, 0.5, 1.0]
    else:
        scale = [0.1, 1.0, 10.0]

    # state error fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e-4, 1)
    
    # ax[1].set_ylim(0, 0.1)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel("One-step data error", fontsize=20)
    ax[1].set_ylabel("Cumul data error", fontsize=20)
    ax[fig_num - 1].set_xlabel("t", fontsize=20)

    for k in range(len(log_list)):
        if bak:
            data_list = torch.load(f'logs/data_bak/error/phase1_test_{log_list[k]}_{ex_name}')
        else:
            data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
        error_1step, error_cul, _, _, _, _ = calMean(data_list)
        error_1step_v, error_cul_v, _, _, _, _ = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')

            ax[0].fill_between(t_nn, error_1step_v[0][scale_k[j]], error_1step_v[1][scale_k[j]], alpha=0.2)
            ax[1].fill_between(t_nn, error_cul_v[0][scale_k[j]], error_cul_v[1][scale_k[j]], alpha=0.2)
            
            ax[0].legend()
    
    if bak:
        plt.savefig(f'logs/pics_bak/error/phase1_state_{fig_name}_{ex_name}.jpg')
    else:
        plt.savefig(f'logs/pics/error/phase1_state_{fig_name}_{ex_name}.jpg')
    
    # fig setting
    fig_num = 2
    fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    ax = ax.flatten()
    for i in range(fig_num):
        ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
        ax[i].grid(True, lw=0.4, ls="--", c=".50")
        # ax[i].set_xlim(0, t_nn[-1])
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e-4, 1)
        
    ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    ax[0].set_ylabel(r'Cul $C_D$ error', fontsize=20)
    ax[1].set_ylabel(r'Cul $C_L$ error', fontsize=20)
    ax[fig_num-1].set_xlabel("t", fontsize=20)
    
    for k in range(len(log_list)):
        if bak:
            data_list = torch.load(f'logs/data_bak/error/phase1_test_{log_list[k]}_{ex_name}')
        else:
            data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
        _, _, _, _, error_Cd_cul, error_Cl_cul = calMean(data_list)
        _, _, _, _, error_Cd_cul_v, error_Cl_cul_v = calVar(data_list)
        
        for j in range(len(scale_k)):
            ax[0].plot(t_nn, error_Cd_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            ax[1].plot(t_nn, error_Cl_cul[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            
            ax[0].fill_between(t_nn, error_Cd_cul_v[0][scale_k[j]], error_Cd_cul_v[1][scale_k[j]], alpha=0.2)
            ax[1].fill_between(t_nn, error_Cl_cul_v[0][scale_k[j]], error_Cl_cul_v[1][scale_k[j]], alpha=0.2)
            
            ax[0].legend()
            
    if bak:
        plt.savefig(f'logs/pics_bak/error/phase1_culcoef_{fig_name}_{ex_name}.jpg')
    else:
        plt.savefig(f'logs/pics/error/phase1_culcoef_{fig_name}_{ex_name}.jpg')
    
    # # 1 step coef fig setting
    # fig_num = 2
    # fig, ax = plt.subplots(nrows=fig_num, ncols=1, figsize=(15,12), dpi=200)
    # ax = ax.flatten()
    # for i in range(fig_num):
    #     ax[i] = plt.subplot2grid((fig_num, 1), (i, 0))
    #     ax[i].grid(True, lw=0.4, ls="--", c=".50")
    #     # ax[i].set_xlim(0, t_nn[-1])
    #     ax[i].set_yscale('log')
    #     ax[i].set_ylim(1e-4, 1e-1)
        
    # ax[0].set_title("Error/Loss in Different Scales", fontsize=20)
    # ax[0].set_ylabel(r'One-step $C_D$ error', fontsize=20)
    # ax[1].set_ylabel(r'One-step $C_L$ error', fontsize=20)
    # ax[fig_num-1].set_xlabel("t", fontsize=20)
    
    # for k in range(len(log_list)):
    #     data_list = torch.load(f'logs/data/error/phase1_test_{log_list[k]}_{ex_name}')
    #     _, _, error_Cd_1step, error_Cl_1step, _, _ = calMean(data_list)
    #     _, _, error_Cd_1step_v, error_Cl_1step_v, _, _ = calVar(data_list)
        
    #     for j in range(len(scale_k)):
    #         ax[0].plot(t_nn, error_Cd_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
    #         ax[1].plot(t_nn, error_Cl_1step[scale_k[j]], label = f'{log_list[k]}, scale={scale[scale_k[j]]}')
            
    #         # ax[0].fill_between(t_nn, error_Cd_1step_v[0][scale_k[j]], error_Cd_1step_v[1][scale_k[j]], alpha=0.2)
    #         # ax[1].fill_between(t_nn, error_Cl_1step_v[0][scale_k[j]], error_Cl_1step_v[1][scale_k[j]], alpha=0.2)
            
    #         ax[0].legend()
            
    # plt.savefig(f'logs/pics/error/phase1_coef_1step_{fig_name}_{ex_name}.jpg')

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
    
def animate_field(data, name, file_name):
    nt = data.shape[0]
    
    x = np.arange(256) / 256 * 2.2
    y = np.arange(64) / 64 * 0.41
    y, x = np.meshgrid(y, x)
    xl, xh  = np.min(x), np.max(x)
    yl, yh = np.min(y), np.max(y)

    figsizer=10
    fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
    ax.axis('equal')
    # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
    ax.set(xlim=(xl, xh), ylim=(yl, yh))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    u, v = [data[:, :, :, i] for i in range(2)]
    w = u**2 + v**2

    def animate(i):
        ax.clear()
        ax.quiver(x, y, u[i], v[i], w[i])
        # ax.plot_surface(x, y, Lpde_obs[i, :, :, 0])
        # ax.plot(x[i], y[i])
        
    print(f'generate anime {name}')
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics/output/{name}_{file_name}_2D.gif')
    
    
def animate2D(data, name, file_name):
    nt = data.shape[0]
    
    x = np.arange(256) / 256 * 2.2
    y = np.arange(64) / 64 * 0.41
    y, x = np.meshgrid(y, x)
    xl, xh  = np.min(x), np.max(x)
    yl, yh = np.min(y), np.max(y)

    figsizer=10
    fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
    ax.axis('equal')
    # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
    ax.set(xlim=(xl, xh), ylim=(yl, yh))
    ax.set_title(f'{name} {file_name}')

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')

    def animate(i):
        ax.clear()
        # print(data[i].shape)
        # ax.quiver(x, y, u[i], v[i], w[i])
        ax.contourf(x, y, data[i], 200, cmap='jet')
        # ax.colorbar()
        # ax.plot_surface(x, y, Lpde_obs[i, :, :, 0])
        # ax.plot(x[i], y[i])
        
    print(f'generate anime {name}')
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_bak/output/{name}_{file_name}_2D.gif')

def animate3D(data, name, file_name, zlim = 100):
    nt = data.shape[0]
    
    x = np.arange(256) / 256 * 2.2
    y = np.arange(64) / 64 * 0.41
    y, x = np.meshgrid(y, x)
    xl, xh  = np.min(x), np.max(x)
    yl, yh = np.min(y), np.max(y)

    # figsizer=10
    # fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
    # ax.axis('equal')
    # # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
    # ax.set(xlim=(xl, xh), ylim=(yl, yh))

    fig = plt.figure(dpi=400)
    ax = plt.axes(projection='3d')
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([2.2, 0.41, 1, 2.2]))
    ax.set_title(f'{name} {file_name}')
    
    u, v = [data[:, :, :, i] for i in range(2)]
    w = u**2 + v**2

    def animate(i):
        ax.clear()
        # ax.quiver(x, y, u[i], v[i], w[i])
        ax.set_zlim(0, zlim)
        ax.plot_surface(x, y, w[i], cmap='rainbow')
        # ax.plot(x[i], y[i])
        
    print(f'generate anime {name}')
    myAnimation = animation.FuncAnimation(fig, animate, frames=np.arange(nt), interval=1, repeat=False)
    myAnimation.save(f'logs/pics_bak/output/{name}_{file_name}.gif')