from Cylinder_Rotation_Env_utils import *
from fenics import *
import numpy as np
sim = Cylinder_Rotation_Sim()

sim.do_simulation(ang_vel=20)
aa = sim.generate_init_state()

sim.set_state_funcval(aa[0], aa[1])

sim.set_state_vector(sim.init_state_1_vector.vector())

plot(sim.solver.sol.sub(0), title='u')
plot(sim.solver.sol.sub(1), title='p')

mesh = sim.solver.geometry.mesh
coor = mesh.coordinates()

u = sim.solver.sol.sub(0)
p = sim.solver.sol.sub(1)

p( coor[0])
p( coor[1])
p( coor[2])

p.vector().get_local().shape
p.compute_vertex_values(mesh).shape


# v2d = vertex_to_dof_map(sim.solver.function_space.V  )

vertex_values = p.compute_vertex_values()

self.params = {'dtr': 0.1,
                                    'T': 1,
                                    'rho_0': 1,
                                    'mu' : 1/1000,
                                    'traj_max_T': 20,
                                    'dimx': 100,
                                    'dimy': 100,
                                    'min_w': -1,
                                    'max_w': 1,
                                    'min_velocity': -1,
                                    'max_velocity': 1,
                                    'min_x' : 0, 
                                    'max_x' : 2.2, 
                                    'min_y' : 0, 
                                    'max_y' : 0.41, 
                                    # 'W' = 0.41,
                                    'r' : 0.05, 
                                    'center':(0.2, 0.2)
                                    }

self = sim
gsx = self.params['dimx']
gsy = self.params['dimy']



xs = np.linspace(self.solver.geometry.min_x, self.solver.geometry.max_x, gsx)
ys = np.linspace(self.solver.geometry.min_y, self.solver.geometry.max_y, gsy)
mx, my = np.meshgrid(xs, ys)
grids = np.stack((mx, my), 2)
self.grids = grids
self.meshgrid = [mx, my]


out = np.zeros((*grids.shape[:2],3))

from joblib import Parallel, delayed

from time import sleep
for _ in range(100):
   sleep(.2)

[sleep(.2) for _ in range(100)]

from joblib import Parallel
r = Parallel(n_jobs=20, verbose=100)(delayed(sleep)(.2) for _ in range(100)) 

self.solver.sol([0,0])


out = np.zeros((*grids.shape[:2],3))
for i in range(grids.shape[0]):
    for j in range(grids.shape[1]):
        xy = grids[i, j]
        try:
            out[i, j] = self.solver.sol(xy)
        except RuntimeError:
            out[i, j] = 0


def to_grid(self,xy):
    return self.solver.sol(xy)


Parallel(n_jobs=4)(delayed(to_grid)(self,xy) for xy in self.grids[0,:10])

Parallel(n_jobs=1)(delayed(to_grid)(i**2) for i in range(10))