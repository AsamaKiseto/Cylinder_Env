from CFD import MyGeometry, MyFunctionSpace, MySolver   
import copy
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import  cm
import matplotlib.tri as tri
set_log_level(30)

class Cylinder_Rotation_Sim:
    def __init__(self,  params):

        min_x, max_x, min_y, max_y = params['min_x'], params['max_x'], params['min_y'], params['max_y']
        r, center = params['r'], params['center']
        dt = params['dtr'] * params['T']
        geometry = MyGeometry(min_x = min_x, max_x = max_x, min_y = min_y,max_y = max_y, r = r, center=center)
        function_space = MyFunctionSpace(geometry, )
        solver = MySolver(geometry, function_space, params=params)
        
        # generate geometry and function space
        solver.geometry.generate()
        solver.function_space.generate()
        
        # set class members
        self.params = params
        self.solver = solver
        self.geometry = geometry
        self.function_space = function_space
        # self.init_state_0 = se 
        self.solver.fixed_boundary_conditions()
        self.solver.changeable_boundary_conditions(ang_vel=0.0000, )
        self.solver.generate_bc()
        # self.solver.changeable_boundary_conditions(ang_vel=0, )
        print("start init_solve")
        self.solver.init_solve()
        print("end init_solve")
        self.init_sol_1 = Function(self.function_space.V)
        self.init_sol_1.vector()[:] = solver.sol_1.vector()
        self.init_sol_n = Function(self.function_space.V)
        self.init_sol_n.vector()[:] = solver.sol_n.vector()
        self.solver.generate_sol_var(dt)
        # self.init_state_1_vector =   solver.sol.vector()
    
    def set_state_funcval(self, initu, initp):
        assign(self.solver.sol, [initu, initp])
    
    def set_init_vector(self, init_sol_1, init_sol_n):
        self.init_sol_1.vector()[:] = init_sol_1
        self.init_sol_n.vector()[:] = init_sol_n
    
    def save_sol(self):
        self.log_sol_1 = self.solver.sol_1.vector()
        self.log_sol_n = self.solver.sol_n.vector()

    def reset_state_vector(self):
        self.solver.sol.vector()[:] = self.init_sol_n.vector()
        self.solver.sol_1.vector()[:] = self.init_sol_1.vector()
        self.solver.sol_n.vector()[:] = self.init_sol_n.vector()

    def generate_init_state(self, init_state=(  ('0', '0'), '0')):
                                                # (Constant(0)) )):
        initu, initp = init_state
        initu = interpolate(Expression( (initu[0], initu[1] ), degree=2), self.solver.function_space.V.sub(0).collapse())
        initp = interpolate(Expression(initp, degree=2), self.solver.function_space.V.sub(1).collapse())
  
        return (initu, initp)
    
    def do_simulation(self, ang_vel=0 ):
        self.solver.changeable_boundary_conditions(ang_vel=ang_vel, )
        self.solver.generate_bc()
        self.solver.generate_solver()
        # self.solver.generate_sol_var()
        self.solver.solve_step()

    def get_state(self):
        return self.solver.sol.vector()

    # def to_grid(self,xy):
    #     return self.solver.sol(xy)

    # if mode == 'node':
            # current_obs = self.solver.sol.vector().get_local()
 
    def get_observation(self, mode):
        if mode == 'vertex':
            vertex_obs = self.solver.sol.compute_vertex_values()
            nv = self.solver.geometry.num_vertices
             
            self.current_obs = np.concatenate([self.solver.geometry.mesh_coor, np.array([vertex_obs[i * nv: (i + 1) * nv] for i in range(3)]).transpose()], axis=1)
            self.observation_mode = 'vertex'

        elif mode == 'grid':
            out = np.zeros((*self.grids.shape[:2],3))
            for i in range(self.grids.shape[0]):
                for j in range(self.grids.shape[1]):
                    xy = self.grids[i, j]
                    center = self.params['center']
                    r = self.params['r']
                    if np.linalg.norm(xy - center) > r:
                        out[i, j] = self.solver.sol(xy)         # sol & sol_n
                    else:
                        out[i, j] = 0
            self.current_obs = np.concatenate([self.grids, out], axis=-1)
            # self.current_obs = out
            self.observation_mode = 'grid'

        return self.current_obs
             
    # plt.quiver(self.current_obs[:,0],self.current_obs[:,1],self.current_obs[:,2],self.current_obs[:,3],self.current_obs[:,2]**2 + self.current_obs[:,3]**2 )    
    def generate_grid(self):
        gsx = self.params['dimx']
        gsy = self.params['dimy']
        xs = np.linspace(self.solver.geometry.min_x, self.solver.geometry.max_x, gsx)
        ys = np.linspace(self.solver.geometry.min_y, self.solver.geometry.max_y, gsy)
        # mx, my = np.meshgrid(xs, ys)
        my, mx = np.meshgrid(ys, xs)

        grids = np.stack((mx, my), 2)
        self.grids = grids
        self.meshgrid = [mx, my]

    def get_mask(self, mode):
        if mode == 'raw':
            return (self.zero_mask, self.non_zero_mask)
        elif mode == 'argument':
            return (self.zero_mask_argument, self.non_zero_mask_argument)

    def cal_mask(self):
        grids = self.grids
        out = np.zeros(grids.shape[:2])
        for i in range(grids.shape[0]):
            for j in range(grids.shape[1]):
                xy = grids[i, j]
                try:
                    self.solver.sol(xy)
                except RuntimeError:
                    out[i, j] = 1
 
        out_argument  = out.copy()
        x,y = np.where(out_argument ==1)
        for i in range(x.shape[0]):
            out_argument[x[i], y[i]] = 1
            out_argument[x[i]-1, y[i]] = 1
            out_argument[x[i]+1, y[i]] = 1
            out_argument[x[i], y[i]-1] = 1
            out_argument[x[i], y[i]+1] = 1
        self.zero_mask_argument = out_argument == 1
        self.non_zero_mask_argument = out_argument == 0

        out_raw = out.copy()
        self.zero_mask = out_raw == 1
        self.non_zero_mask = out_raw == 0
    
    def postprocess(self, sol):
        u, p = sol.split()
        mu = self.params['mu']
        ds_circle = Measure("ds", subdomain_data=self.solver.geometry.bndry, subdomain_id=5)
        # Report drag and lift
        n = FacetNormal(sol.function_space().mesh())
        force = -p*n +  mu *dot(grad(u), n)
        F_D = assemble(-force[0]*ds_circle)
        F_L = assemble(-force[1]*ds_circle)

        U_mean = self.params['U_max']* 2/3
        L = 2* self.params['r']
        C_D = 2/(U_mean**2*L)*F_D
        C_L = 2/(U_mean**2*L)*F_L

        # Report pressure difference
        a_1 = Point(0.15, 0.2)
        a_2 = Point(0.25, 0.2)
        try:
            p_diff = p(a_1) - p(a_2)
        except RuntimeError:
            p_diff = 0

        return C_D, C_L, p_diff

    def plot_vel(self, figsizer=10 ):
        x, y, u, v = [self.current_obs[:,i] for i in range(4)]

        # u[self.zero_mask.reshape(-1)] = None
        # v[self.zero_mask.reshape(-1)] = None
        w = u**2 + v**2
        xl, xh  = np.min(x), np.max(x)
        yl, yh = np.min(y), np.max(y)

        fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
        ax.axis('equal')
        # ax.set(xlim=(0, 2.2), ylim=(0, 0.41))
        ax.set(xlim=(xl, xh), ylim=(yl, yh))
        ax.quiver(x, y, u, v, w)
        plt.show()

    def plot_pressure(self, figsizer=10):
        x, y, p = [self.current_obs[:,i] for i in [0,1,4]]
        pressure = copy.deepcopy(p)
        
        if self.observation_mode == 'grid':
            triang = tri.Triangulation(x, y)
            non_zero_mask  = self.non_zero_mask.reshape(-1)
            b = non_zero_mask[triang.triangles.reshape(-1)].reshape(-1,3)
            c = np.where(b.sum(1)==0)[0]
            mask =  np.zeros((triang.triangles.shape[0],))
            mask[c] = 1
            triang.set_mask(mask)
        else:
            xy = self.geometry.mesh_coor
            mesh = self.geometry.mesh
            triang = tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())
        xl, xh  = np.min(x), np.max(x)
        yl, yh = np.min(y), np.max(y)
        fig, ax = plt.subplots(figsize=((xh - xl)*figsizer,(yh-yl)*figsizer))
        ax.axis('equal')
        ax.set(xlim=(xl, xh), ylim=(yl, yh))
        ax.tricontourf(triang, pressure, alpha=0.5, cmap=cm.viridis,  )
        ax.tricontour(triang, pressure, cmap=cm.viridis)  
        plt.show()