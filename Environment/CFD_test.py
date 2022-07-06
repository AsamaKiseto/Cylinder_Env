from CFD import MyGeometry, MyFunctionSpace, MySolver   

from fenics import set_log_level, plot
from fenics import *
geometry = MyGeometry(L = 2.2, W = 0.41, r = 0.05, center=(0.2, 0.2))
function_space = MyFunctionSpace(geometry, )
solver = MySolver(geometry, function_space, params={'dtr': 0.1,
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
                            })


solver.geometry.generate()
solver.function_space.generate()


set_log_level(30)
solver.fixed_boundary_conditions()
solver.changeable_boundary_conditions(ang_vel=0.0000, )
solver.generate_bc()
solver.generate_sol_var()
solver.generate_solver()
solver.init_solve()
solver.solve()
p = solver.sol.split()[1]
plot(p, title='p')
import matplotlib.pyplot as plt
plt.show()
# print([solver.sol(*x) for x in [(0,0), (0,0.1), (0.1,0), (0.1,0.1)] ])

# for i in range(20):
#     solver.solve()

#     plot(solver.sol.split()[0], title='u')


p = solver.sol.split()[1]
u = solver.sol.split()[0]
plot(p, title='p')
plot(u, title='u')

initp = '0'

gp = interpolate(Expression(initp, degree=2), solver.function_space.V.sub(1).collapse())
gu = interpolate(Expression((initp, initp), degree=2), solver.function_space.V.sub(0).collapse())

assign(solver.sol, [gu, gp])

p.assign(gp)
u.assign(gu)

assign(solver.sol, a)

solver.sol.vector()[:] = a