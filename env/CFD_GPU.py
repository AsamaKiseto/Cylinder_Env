from re import A
from fenics import *
import mshr
import numpy as np
import matplotlib.pyplot as plt
import cupy
import cupyx
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import cupyx.scipy.sparse.linalg

mempool = cupy.get_default_memory_pool()
with cupy.cuda.Device(0):
    mempool.set_limit(size=1.2*1024**3)
parameters['linear_algebra_backend'] = 'Eigen'

def tran2SparseMatrix(A):
    row, col, val = as_backend_type(A).data()
    return sps.csr_matrix((val, col, row))

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
        # self.mesh = mshr.generate_mesh(domain, 128)
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
        self.bcu = [self.bc_walls, self.bc_in, self.bc_cylinder]
        self.bcp = [self.bc_out]

    def generate_sol_var(self):
        V = self.function_space.V
        params = self.params
        k  = params['dtr'] * params['T'] #dt
        mu = params['mu']

        # Define trial and test functions
        sol = TrialFunction(V)
        sol_n = Function(V)
        sol_ = Function(V)
        v, q = TestFunctions(V)

        # Define functions for solutions at previous and current time steps
        u, p = split(sol)
        u_n, p_n = sol_n.split(deepcopy=True)
        u_, p_ = sol_.split(deepcopy=True)

        # Define expressions used in variational forms
        U  = 0.5*(u_n + u)
        n  = FacetNormal(self.geometry.mesh)
        f  = Constant((0, 0))
        
        # mu = Constant(mu)
        rho = Constant(1)   # density

        # Define symmetric gradient
        def epsilon(u):
            return sym(nabla_grad(u))

        # Define stress tensor
        def sigma(u, p):
            return 2*mu*epsilon(u) - p*Identity(len(u))

        # Define variational problem for step 1
        F1 = rho*dot((u - u_n) / k, v)*dx \
             + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
             + inner(sigma(U, p_n), epsilon(v))*dx \
             + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
             - dot(f, v)*dx
        a1 = lhs(F1)
        L1 = rhs(F1)

        # Define variational problem for step 2
        a2 = dot(nabla_grad(p), nabla_grad(q))*dx
        L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

        # Define variational problem for step 3
        a3 = dot(u, v)*dx
        L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

        self.L1 = L1
        self.L2 = L2
        self.L3 = L3

        # Assemble matrices
        self.A1 = assemble(a1)
        self.A2 = assemble(a2)
        self.A3 = assemble(a3)
        self.b1 = assemble(L1)
        self.b2 = assemble(L2)
        self.b3 = assemble(L3)
        print("Grid Points:",np.size(self.b1[:]))

        # Apply boundary conditions to matrices
        [bc.apply(self.A1) for bc in self.bcu]
        [bc.apply(self.A2) for bc in self.bcp]

        #Converting to Sparse Matrix
        self.A1 = tran2SparseMatrix(self.A1)
        self.A2 = tran2SparseMatrix(self.A2)
        self.A3 = tran2SparseMatrix(self.A3)
        print(self.A1)
        
        self.sol = sol
        self.sol_n = sol_n
        self.sol_ = sol_
        self.u_, self.p_ = u_, p_
        self.v, self.q = v, q

    def solve_step(self):
        
        self.b1 = assemble(self.L1)
        [bc.apply(self.b1) for bc in self.bcu]
        b1 = self.b1[:]
        As1 = cupyx.scipy.sparse.csr_matrix(self.A1)
        print(As1)
        print(b1)
        # bs1 = cupy.array(self.b1)
        bs1 = cupy.array(b1)
        self.u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As1, bs1)[:1][0])
        print('u_', self.u_.vector()[:])

        self.b2 = assemble(self.L2)
        [bc.apply(self.b2) for bc in self.bcp]
        b2 = self.b2[:]
        As2 = cupyx.scipy.sparse.csr_matrix(self.A2)
        # bs2 = cupy.array(self.b2)
        bs2 = cupy.array(b2)
        self.p_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As2, bs2)[:1][0])

        self.b3 = assemble(self.L3)
        As3 = cupyx.scipy.sparse.csr_matrix(self.A3)
        b3 = self.b3[:]
        # bs3 = cupy.array(self.b3)
        bs3 = cupy.array(b3)
        self.u_.vector()[:] = cupy.asnumpy(cupyx.scipy.sparse.linalg.lsqr(As3, bs3)[:1][0])

        assign(self.sol_, [self.u_, self.p_])
        self.sol_n.assign(self.sol_)

        # print('u max:', self.u_.vector().max())
        # print('p max:', self.p_.vector().max())
        # print('sol max', self.sol_.vector().max())

    def init_solve(self):
        n_ts = 10
        for _ in range(n_ts):
            self.solve_step()