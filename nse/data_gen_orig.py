from CFD_data import MyGeometry, MyFunctionSpace, MySolver   
import numpy as np
import torch
from timeit import default_timer
from fenics import set_log_level, plot
from fenics import *

geometry = MyGeometry(max_x = 2.2, max_y = 0.41, r = 0.05, center=(0.2, 0.2))
function_space = MyFunctionSpace(geometry, )
solver = MySolver(geometry, function_space, params={'dtr': 0.015,'dt' : 0.05,
                            'T': 2.01,'U_max':1.5,'r' : 0.05,
                            'rho_0': 1,
                            'mu' : 1/1000,
                            'traj_max_T': 20,
                            'dimx': 100,
                            'dimy': 100,
                            'min_w': -1,
                            'max_w': 1,
                            'min_velocity': -1,
                            'max_velocity': 1,
                            })

set_log_level(30)

geometry.generate()
solver.function_space.generate()
solver.fixed_boundary_conditions()
solver.generate_sol_var()

def data_gen(ang_vel):
    # random angel velocity
    solver.changeable_boundary_conditions(ang_vel, )

    solver.generate_bc()
    solver.generate_solver()

    solver.init_solve()
    solver.solve_start()
    solver.solve_step1()

    data_v1,data_v2,data_p = solver.datav1,solver.datav2,solver.datap

    data_v1 = torch.Tensor(data_v1.squeeze()).unsqueeze(-1)
    data_v2 = torch.Tensor(data_v2.squeeze()).unsqueeze(-1)
    data_p = torch.Tensor(data_p.squeeze()).unsqueeze(-1)

    return torch.cat((data_v1, data_v2, data_p), dim=-1)

if __name__=='__main__':
    # data param
    N0 = 25     # N0 set of data
    ny = solver.n_y
    nx = solver.n_x
    nt = solver.nt
    print("N0: {}, ny: {}, nx: {}, nt: {}".format(N0, ny, nx, nt))
    
    data = torch.empty(N0, nt, ny, nx, 3)
    Cd = []
    Cl = []
    ang_vel = np.random.rand(N0)*0.5
    print(ang_vel)

    for i in range(N0):

        t1 = default_timer()

        print('Generate Data # {}'.format(i))
        data[i] = data_gen(ang_vel[i])
        Cd.append(solver.list_D)
        Cl.append(solver.list_L)

        t2 = default_timer()
        print("Used time:{}".format(t2-t1))
    
    torch.save([data, Cd, Cl, torch.Tensor(ang_vel)], 'home/fenics/shared/nse/data/nse_data_dt_0.05_T_2')
    # torch.save([data, torch.Tensor(ang_vel)], 'nse_control_samples1')