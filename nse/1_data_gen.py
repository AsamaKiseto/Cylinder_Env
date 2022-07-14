from env.Cylinder_Rotation_Env import Cylinder_Rotation_Env
import numpy as np
import torch
from timeit import default_timer

# env init
env = Cylinder_Rotation_Env(params={'dtr': 0.01, 'T': 5, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 128, 'dimy': 64,
                                    'min_x' : 0,  'max_x' : 2.2, 
                                    'min_y' : 0,  'max_y' : 0.41, 
                                    'r' : 0.05,  'center':(0.2, 0.2),
                                    'min_w': -1, 'max_w': 1,
                                    'min_velocity': -1, 'max_velocity': 1,
                                    'U_max': 1.5, })

print(env.params)

# param setting
N0 = 25     # N0 set of data
dt = env.params['dtr'] * env.params['T']
nt = int(- (env.params['T'] // -dt))
nx = env.params['dimx']
ny = env.params['dimy']
print(f'dt: {dt} | nt: {nt}')

# data generate
obs = np.zeros((N0, nt, nx, ny, 3))
print(f'state_data_size :{obs.shape}')
C_D, C_L, reward = np.zeros((N0, nt)), np.zeros((N0, nt)), np.zeros((N0, nt))
ang_vel = np.random.rand(N0, nt)
for k in range(N0):
    start = default_timer()
    obs[k] = env.reset()
    
    ang_v = ang_vel[k]
    
    for i in range(nt):
        # obs, reward, C_D, C_L, episode_over, _ = env.step(ang_vel[i])
        obs[k, i], reward[k, i], C_D[k, i], C_L[k, i] = env.step(ang_v[i])
    
    end = default_timer()

    print(f'# {k} | time: {end-start}')
    # print(f'ang_vel: {ang_v}')
    # print(f'reward :{reward[k]}')

# np to tensor
obs_tensor = torch.Tensor(obs)
reward_tensor = torch.Tensor(reward)
C_D_tensor = torch.Tensor(C_D)
C_L_tensor = torch.Tensor(C_L)
ang_vel_tensor = torch.Tensor(ang_vel)

# save data
torch.save([obs_tensor, reward_tensor, C_D_tensor, C_L_tensor, ang_vel_tensor], './data/nse_data_N0_{}_dtr_{}_T_{}'.format(N0, env.params['dtr'], env.params['T']))