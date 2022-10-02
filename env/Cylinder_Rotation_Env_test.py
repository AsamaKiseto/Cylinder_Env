from Cylinder_Rotation_Env import Cylinder_Rotation_Env
import matplotlib.pyplot as plt
from fenics import plot

env = Cylinder_Rotation_Env(params={'dt': 0.1, 'rho_0': 1, 'mu' : 1/1000,
                                    'traj_max_T': 20, 'dimx': 128, 'dimy': 64,
                                    'min_x' : 0,  'max_x' : 2.2, 
                                    'min_y' : 0,  'max_y' : 0.41, 
                                    'r' : 0.05,  'center':(0.2, 0.2),
                                    'min_w': -1, 'max_w': 1,
                                    'min_velocity': -1, 'max_velocity': 1,
                                    'U_max': 1.5, })

print(env.params)
obs = env.reset()
ob, reward, episode_over, _ = env.step(-500)
print('ob: {} | reward: {} '.format(ob.shape, reward))

env.sim.get_observation('vertex')
env.sim.plot_vel()
env.sim.plot_pressure()
