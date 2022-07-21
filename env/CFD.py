from fenics import *
import matplotlib.pyplot as plt 
import mshr
import numpy as np

class MyGeometry:
    def __init__(self,  min_x = 0, max_x = 2.2, min_y = 0,max_y = 0.41, r = 0.05, center=(0.2, 0.2), params=None):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        # self.W = W
        self.r = r
        self.params = params
        # self.c
        self.center = Point(center[0], center[1])
    
    def generate(self, params=None):

        channel = mshr.Rectangle(Point(self.min_x, self.min_y), Point(self.max_x, self.max_y))
        # channel = mshr.Rectangle(Point(0, 0), Point(self.L, self.W))

        cylinder = mshr.Circle(self.center, self.r)
        domain = channel  - cylinder
        self.mesh = mshr.generate_mesh(domain, 64)
        bndry = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        for f in facets(self.mesh):
            mp = f.midpoint()
            if near(mp[0], self.min_x):  # inflow
                bndry[f] = 1
            elif near(mp[0], self.max_x):  # outflow
                bndry[f] = 2
            elif near(mp[1], self.max_y) or near(mp[1], self.min_y):  # walls
                bndry[f] = 3
            elif mp.distance(self.center) <= self.r:  # cylinder
                bndry[f] = 5
        self.bndry = bndry
        self.mesh_coor = self.mesh.coordinates()
        self.num_vertices = self.mesh.num_vertices()


class MyFunctionSpace:
    def __init__(self, geometry, params=None):
        self.geometry = geometry
        self.params = params
    
    def generate(self, params=None):
 
        E1 = FiniteElement("P", self.geometry.mesh.ufl_cell(), 1)
        E2 = VectorElement("P", self.geometry.mesh.ufl_cell(), 2)
        V = FunctionSpace(self.geometry.mesh, MixedElement([E2, E1]))
        self.V = V

class MySolver:
    def __init__(self, geometry, function_space, params=None):
        self.geometry = geometry
        self.function_space = function_space
        self.params = params
        self.time = 0
        self.u_in = Expression((f"4.0*U*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"),
                      degree=2, U=self.params['U_max'])
        # self.u_in = Expression(('0.001', '0.0'), degree=2)
        self.center = np.array([0.2,0.2])
        self.r = 0.05
    
    def fixed_boundary_conditions(self):
        self.bc_walls = DirichletBC(self.function_space.V.sub(0), (0, 0), self.geometry.bndry, 3)
        self.bc_in = DirichletBC(self.function_space.V.sub(0), self.u_in, self.geometry.bndry, 1)
        self.bc_out = DirichletBC(self.function_space.V.sub(1), (0), self.geometry.bndry, 2)

    def changeable_boundary_conditions(self, ang_vel):
        # print('ang velocity :{}'.format(ang_vel))
        u_c = Expression((f" {ang_vel} * {self.geometry.r} * ( x[1] - {self.geometry.center[1]} )/ {self.geometry.r}  ", f" -1* {ang_vel} * {self.geometry.r} * ( x[0] - {self.geometry.center[0]} )/ {self.geometry.r}  "), degree=2)
        self.bc_cylinder = DirichletBC(self.function_space.V.sub(0), u_c, self.geometry.bndry, 5)

    def generate_bc(self):
        self.bcs = [self.bc_walls, self.bc_in, self.bc_cylinder]

    def generate_sol_var(self):
        params = self.params
        dt  = params['dtr'] * params['T']
        mu = params['mu']
        V = self.function_space.V
        sol = Function(V)
        u, p = split(sol)
        sol_n = Function(V)
        u_n, p_n = split(sol_n)
        sol_1 =  Function(V)
        u_1, p_1 = split(sol_1)
        
        u_t,   p_t = TestFunctions(V)
        theta = 0.5
        
        F = (Constant(3.0/dt/2.0)*inner(u,u_t)-Constant(4.0/dt/2.0)*inner(u_n,u_t)
            +Constant(1.0/dt/2.0)*inner(u_1,u_t)
            +Constant(2.0)*dot(dot(grad(u), u_n), u_t)
            -dot(dot(grad(u), u_1), u_t)
            + mu*inner(grad(u), grad(u_t))
            - p*div(u_t)
            + p_t*div(u)
            )*dx

        F1 = ( Constant(1/dt)*inner(u - u_n, u_t)
                + Constant(theta)*mu*inner(grad(u), grad(u_t))
                + Constant(theta)*dot(dot(grad(u), u), u_t)
                + Constant(1-theta)*mu*inner(grad(u_n), grad(u_t))
                + Constant(1-theta)*inner(dot(grad(u_n), u_n), u_t)
                - p*div(u_t)
                + p_t*div(u)
                )*dx
        
        J = derivative(F, sol)
        J1 =  derivative(F1, sol)
        self.F = F
        self.J = J
        self.F1 = F1
        self.J1 = J1

        self.sol = sol
        self.sol_n = sol_n
        self.sol_1 = sol_1
        self.u_t, self.p_t = u_t, p_t

    def generate_solver(self):
        params = self.params
 
        problem = NonlinearVariationalProblem(self.F, self.sol, self.bcs, self.J)
        problem1 = NonlinearVariationalProblem(self.F1, self.sol, self.bcs, self.J1)
        solver = NonlinearVariationalSolver(problem)
        solver1 = NonlinearVariationalSolver(problem1)
        solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        solver1.parameters['newton_solver']['linear_solver'] = 'mumps'
        self.solver = solver
        self.solver1 = solver1
        self.problem1 = problem1

        self.problem = problem
        # self.sol = sol
        # self.sol_n = sol_n
    
    def solve_start(self):
        params = self.params
        dt = params['dtr'] * params['T']
        T  = params['T']
        n_ts = int(-(T // -dt))
        self.sol_n.vector()[:] = self.sol.vector()
        self.sol_1.vector()[:] = self.sol.vector()
        self.solver1.solve()
        self.sol_n.vector()[:] = self.sol.vector()
        self.time += dt 
        
    # def solve(self):
    #     params = self.params
    #     dt = params['dt'] * params['T']
    #     T  = params['T']
    #     n_ts = int(-(T // -dt))
        
    #     for i_step in range(n_ts):
    #         self.solver.solve()
    #         self.sol_1.vector()[:] = self.sol_n.vector()
    #         self.sol_n.vector()[:] = self.sol.vector()
        
    def set_sol_value(self, sol_value):
        self.sol.vector()[:] = sol_value

    def init_solve(self):       # dt T param
        dt = 1e-6
        T  = 1e-5
        n_ts = int(-(T // -dt))
        # n_ts = 1
        
        for i_step in range(n_ts):

            self.sol_n.vector()[:] = self.sol.vector()
            self.sol_1.vector()[:] = self.sol.vector()
            self.solver1.solve()
            self.sol_n.vector()[:] = self.sol.vector()
            self.time += dt

    def solve_step(self):
        self.solver.solve()
        self.sol_1.vector()[:] = self.sol_n.vector()
        self.sol_n.vector()[:] = self.sol.vector()

        params = self.params
        dt = params['dtr'] * params['T']
        T  = params['T']
        # n_ts = int(-(T // -dt))
        # for i in range(n_ts):
        #     self.solver.solve()
        #     self.sol_1.vector()[:] = self.sol_n.vector()
        #     self.sol_n.vector()[:] = self.sol.vector()

        #     # Plot solution
        #     u_, p_ = split(self.sol)
        #     # plot(self.sol, title='State')
        #     # plt.show()
        #     plot(u_, title='Velocity')
        #     plt.show()
        #     plot(p_, title='Pressure')
        #     plt.show()

            
        
    