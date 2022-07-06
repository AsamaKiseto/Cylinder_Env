from fenics import *
import mshr


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
            elif near(mp[1], self.max_y) or near(mp[1], self.max_y):  # walls
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
        self.u_in = Expression((f"4.0*U*x[1]*(0.41 - x[1])/(0.41*0.41)", "0.0"),
                      degree=2, U=self.params['U_max'])
        # self.u_in = Expression(('0.001', '0.0'), degree=2)
    
    def fixed_boundary_conditions(self):
        print(self.function_space.V)
        self.bc_walls = DirichletBC(self.function_space.V.sub(0), (0, 0), self.geometry.bndry, 3)
        
        # bc_cylinder = DirichletBC(V.sub(0), (0, 0), Leftv())
        # bc_in = DirichletBC(V.sub(0), (0.001 ,0 ), bndry, 1)
        self.bc_in = DirichletBC(self.function_space.V.sub(0), self.u_in, self.geometry.bndry, 1)
        self.bc_out = DirichletBC(self.function_space.V.sub(1), (0), self.geometry.bndry, 2)
        # bcs = [ bc_cylinder, bc_walls, bc_in]
    def changeable_boundary_conditions(self, ang_vel):
        # print(self.geometry.r, ang_vel)  
        # assert 1==2
        # print(    f"{ang_vel} * {self.geometry.r}"  )
        
        u_c = Expression((f" {ang_vel} * {self.geometry.r} * ( x[1] - {self.geometry.center[1]} ) ", f" -1* {ang_vel} * {self.geometry.r} * ( x[0] - {self.geometry.center[0]} ) "), degree=2)
        self.bc_cylinder = DirichletBC(self.function_space.V.sub(0), u_c, self.geometry.bndry, 5)

    def generate_bc(self):
        self.bcs = [self.bc_walls, self.bc_in, self.bc_cylinder]
    
    def generate_sol_var(self):
        params = self.params
        dt = params['dtr'] * params['T']
        mu = params['mu']
        V = self.function_space.V
        sol = Function(V)
        u , p = split(sol)
        sol_n = Function(V)
        u_n,   p_n = split(sol_n)
        u_t,   p_t = TestFunctions(V)
        
        theta = 0.5
        
        F = ( Constant(1/dt)*inner(u - u_n, u_t)
                + Constant(theta)*mu*inner(grad(u), grad(u_t))
                + Constant(theta)*dot(dot(grad(u), u), u_t)
                + Constant(1-theta)*mu*inner(grad(u), grad(u_t))
                + Constant(1-theta)*inner(dot(grad(u_n), u_n), u_t)
                - p*div(u_t)
                #   + dot(nabla_grad(p),v)
                + p_t*div(u)
                )*dx
        J = derivative(F, sol)
        self.F = F
        self.J = J
        self.sol = sol
        self.sol_n = sol_n
        self.u_t, self.p_t = u_t, p_t

    def generate_solver(self):
        params = self.params
       
        problem = NonlinearVariationalProblem(self.F, self.sol, self.bcs, self.J)
        solver = NonlinearVariationalSolver(problem)
        solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        self.solver = solver
        self.problem = problem
        # self.sol = sol
        # self.sol_n = sol_n
    
    def solve(self):
        params = self.params
        dt = params['dtr'] * params['T']
        T  = params['T']
 
        n_ts = int(-(T // -dt))
        
        for i_step in range(n_ts):
            # print(t)
            self.sol_n.vector()[:] = self.sol.vector()
            self.solver.solve()
        

    def set_sol_value(self, sol_value):
        self.sol.vector()[:] = sol_value

    def init_solve(self):
        # params = self.params
        dt = 1e-6
        T  = 1e-5
 
        n_ts = int(-(T // -dt))
        
        for i_step in range(n_ts):
            # print(t)
            self.sol_n.vector()[:] = self.sol.vector()
            self.solver.solve()