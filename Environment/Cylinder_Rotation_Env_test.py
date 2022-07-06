from Cylinder_Env import Cylinder_Rotation_Env
import matplotlib.pyplot as plt
from fenics import plot

env = Cylinder_Rotation_Env()
obs = env.reset()
ob, reward, episode_over, _ = env.step(-500)

env.sim.get_observation('vertex')
env.sim.plot_vel()
env.sim.plot_pressure()
