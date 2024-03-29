from fenics import *
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
        self.mesh = mshr.generate_mesh(domain, 128)
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
        self.list_D = []
        self.center = np.array([0.2,0.2])
        self.r = 0.05
        self.list_L = []
        self.n_x = 64
        self.n_y = 32
        self.x_grid = np.linspace(0,2.2,self.n_x)
        self.y_grid = np.linspace(0,0.41,self.n_y)
        self.mesh_X,self.mesh_Y = np.meshgrid(self.x_grid,self.y_grid)
        self.grid = np.zeros((self.n_y,self.n_x,2))
        self.grid[:,:,0] = self.mesh_X
        self.grid[:,:,1] = self.mesh_Y
        self.t_list= []
        self.T = self.params['T']
        self.dt =  self.params['dt']
        self.nt = int(-(self.T // -self.dt))
        self.datav1 = np.zeros((self.nt,self.n_y,self.n_x))
        self.datav2 = np.zeros((self.nt,self.n_y,self.n_x))
        self.datap = np.zeros((self.nt,self.n_y,self.n_x))
    
    def fixed_boundary_conditions(self):
        # print(self.function_space.V)
        self.bc_walls = DirichletBC(self.function_space.V.sub(0), (0, 0), self.geometry.bndry, 3)
        
        # bc_cylinder = DirichletBC(V.sub(0), (0, 0), Leftv())
        # bc_in = DirichletBC(V.sub(0), (0.001 ,0 ), bndry, 1)
        self.bc_in = DirichletBC(self.function_space.V.sub(0), self.u_in, self.geometry.bndry, 1)
        self.bc_out = DirichletBC(self.function_space.V.sub(1), (0), self.geometry.bndry, 2)
        # bcs = [ bc_cylinder, bc_walls, bc_in]
    def changeable_boundary_conditions(self, ang_vel):
        print(self.geometry.r, ang_vel)  
        # assert 1==2
        # print(    f"{ang_vel} * {self.geometry.r}"  )
        #coef = (np.random.rand(2,5)*2-1)/1.0**0.5
        #c0 = coef[0,:3]; s0 = coef[0,3:]
        #c1 = coef[1,:3]; s1 = coef[1,3:]
        #cos = '((x[0] - 0.2) / 0.05)'
        #sin = '((x[1] - 0.2) / 0.05)'
        #g0 = str(c0[0]) + ''.join(['+ pow(' + cos + ', {})*({})'.format(n+1,c) for n,c in enumerate(c0[1:])]) \
        #            + ''.join(['+ pow(' + sin + ', {})*({})'.format(n+1,c) for n,c in enumerate(s0)]) 
        #g1 = str(c1[0]) + ''.join(['+ pow(' + cos + ', {})*({})'.format(n+1,c) for n,c in enumerate(c1[1:])]) \
        #            + ''.join(['+ pow(' + sin + ', {})*({})'.format(n+1,c) for n,c in enumerate(s1)])
        #u_c = Expression((g0,g1),degree = 2)
        #self.bc_cylinder = DirichletBC(self.function_space.V.sub(0), u_c, self.geometry.bndry, 5)
        u_c = Expression((f" {ang_vel} * {self.geometry.r} * ( x[1] - {self.geometry.center[1]} )/ {self.geometry.r}  ", f" -1* {ang_vel} * {self.geometry.r} * ( x[0] - {self.geometry.center[0]} )/ {self.geometry.r}  "), degree=2)
        self.bc_cylinder = DirichletBC(self.function_space.V.sub(0), u_c, self.geometry.bndry, 5)

    def generate_bc(self):
        self.bcs = [self.bc_walls, self.bc_in, self.bc_cylinder]

    def generate_sol_var(self):
        params = self.params
        dt  = params['dt'] 
        mu = params['mu']
        V = self.function_space.V
        sol = Function(V)
        u , p = split(sol)
        sol_n = Function(V)
        u_n,   p_n = split(sol_n)
        sol_1 =  Function(V)
        u_1,p_1 = split(sol_1)
        
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
        dt = params['dt'] 
        T  = params['T']
        n_ts = int(-(T // -dt))
        self.sol_n.vector()[:] = self.sol.vector()
        self.sol_1.vector()[:] = self.sol.vector()
        self.solver1.solve()
        self.sol_n.vector()[:] = self.sol.vector()
        self.time += dt 
        
    def solve(self):
        params = self.params
        dt = params['dt'] * params['T']
        T  = params['T']
 
        n_ts = int(-(T // -dt))
        
        for i_step in range(n_ts):
            # print(t)
            self.solver.solve()
            self.sol_1.vector()[:] = self.sol_n.vector()
            self.sol_n.vector()[:] = self.sol.vector()
        
    def set_sol_value(self, sol_value):
        self.sol.vector()[:] = sol_value

    def init_solve(self):
        # params = self.params
        dt = 2e-7
        T  = 1e-5
        n_ts = int(-(T // -dt))
        
        for i_step in range(n_ts):
            # print(t)
            #self.sol_n.vector()[:] = self.sol.vector()
            #self.solver1.solve()

            self.sol_n.vector()[:] = self.sol.vector()
            self.sol_1.vector()[:] = self.sol.vector()
            self.solver1.solve()
            self.sol_n.vector()[:] = self.sol.vector()
            self.time += dt
            
    def solve_step1(self):
        params = self.params
        dt = params['dt'] 
        T = self.T
        n_ts = int(-(T // -dt))
        self.list_D = []
        self.list_L = []

        print('n:',n_ts)
        for i_step in range(n_ts):
            # print(t)
            
            self.time += dt 
            # if i_step % 10  == 0 :
            #     print('real time is' , self.time)
            self.solver.solve()
            self.sol_1.vector()[:] = self.sol_n.vector()
            self.sol_n.vector()[:] = self.sol.vector()
            self.t_list.append(self.time)
            self.datav1[i_step,:,:],self.datav2[i_step,:,:],self.datap[i_step,:,:] = self.get_mesh_data()

            # Cd & Cl
            bnd = self.geometry.bndry
            u, p = self.sol_n.split()
            mu = self.params['mu']
            ds_circle = Measure("ds", subdomain_data=bnd, subdomain_id=5)

            ## Report drag and lift
            n   = FacetNormal(self.sol_n.function_space().mesh())
            force = -p*n +  mu *dot(grad(u), n)
            F_D = assemble(-force[0]*ds_circle)
            F_L = assemble(-force[1]*ds_circle)

            U_mean = self.params['U_max']* 2/3
            L = 2* self.params['r']
            C_D = 2/(U_mean**2*L)*F_D
            C_L = 2/(U_mean**2*L)*F_L

            self.list_D.append(C_D)
            self.list_L.append(C_L)

    def solve_step2(self):
        params = self.params
        dt = params['dt'] 
        T =  1.0
        n_ts = int(-(T // -dt))
        print('n:',n_ts)
        for i_step in range(n_ts):
            # print(t)
            self.time += dt 
            if i_step % 10  == 0 :
                print('real time is' , self.time)
            self.solver.solve()
            self.sol_1.vector()[:] = self.sol_n.vector()
            self.sol_n.vector()[:] = self.sol.vector()
    
    def postprocess(self, sol):
        u, p = sol.split()
        mu = self.params['mu']
        ds_circle = Measure("ds", subdomain_data=bnd, subdomain_id=5)
        # Report drag and lift
        n = FacetNormal(sol.function_space().mesh())
        force = -p*n +  mu *dot(grad(u), n)
        F_D = assemble(-force[0]*ds_circle)
        F_L = assemble(-force[1]*ds_circle)

        U_mean = self.params['U_max']* 2/3
        L = 2* self.params['r']
        C_D = 2/(U_mean**2*L)*F_D
        C_L = 2/(U_mean**2*L)*F_L
        return C_D , C_L

    def simulation(self,bd):
        params = self.params
        dt = params['dt'] 
        T =  2
        n_ts = int(-(T // -dt))
        bnd = self.geometry.bndry
        for i_step in range(n_ts):
            # print(t)
            self.time += dt 
            if i_step % 10  == 0 :
                print('real time is' , self.time)
            self.solver.solve()
            self.sol_1.vector()[:] = self.sol_n.vector()
            self.sol_n.vector()[:] = self.sol.vector()

            #C_D,C_L = self.postprocess(self.sol_n,bnd)
            u, p = self.sol_n.split()
            mu = self.params['mu']
            ds_circle = Measure("ds", subdomain_data=bd, subdomain_id=5)

            ## Report drag and lift
            n   = FacetNormal(self.sol_n.function_space().mesh())
            force = -p*n +  mu *dot(grad(u), n)
            F_D = assemble(-force[0]*ds_circle)
            F_L = assemble(-force[1]*ds_circle)

            U_mean = self.params['U_max']* 2/3
            L = 2* self.params['r']
            C_D = 2/(U_mean**2*L)*F_D
            C_L = 2/(U_mean**2*L)*F_L

            self.list_D.append(C_D)
            self.list_L.append(C_L)
    
    def get_mesh_data(self):
        data_v1 = np.zeros((self.n_y,self.n_x))
        data_v2 = np.zeros((self.n_y,self.n_x))
        data_p = np.zeros((self.n_y,self.n_x))
        u, p = self.sol_n.split()
        for i in range(self.n_x):
            for j in range(self.n_y):
                if np.linalg.norm(self.grid[j,i,:] - self.center) <= self.r:
                    data_v1[j,i] = 0
                    data_v2[j,i] = 0
                    data_p[j,i] = 0
                else :
                    u_val = u(self.grid[j,i,:])
                    data_v1[j,i] = u_val[0]
                    data_v2[j,i] = u_val[1]
                    data_p[j,i] = p(self.grid[j,i,:])
        return data_v1,data_v2,data_p






