from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from fenics import set_log_level, plot

import torch
from functools import partial



'''
""" The exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(pi*x)*np.cos(pi*y)*np.exp(-pi**2/8*t)
'''



class MyGeometry:
    def __init__(self,  min_x = 0.0, max_x =1.0 , min_y = 0.0, max_y = 1.0, params=None):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        # self.W = W
        self.params = params
    
    def generate(self):
        self.nx = self.params['dimx']
        self.ny = self.params['dimy']
        self.mesh = RectangleMesh(Point(self.min_x,self.min_y),Point(self.max_x, self.max_y), self.nx - 1, self.ny - 1)

        bndry = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1)
        for f in facets(self.mesh):
            mp = f.midpoint()
            if near(mp[0], self.min_x):  #left
                bndry[f] = 1
            elif near(mp[0], self.max_x):  # right
                bndry[f] = 2
            elif near(mp[1], self.max_y):  #top
                bndry[f] = 3
            elif near(mp[1], self.min_y): #bottom
                bndry[f] = 4  
  
        self.bndry = bndry
        self.mesh_coor = self.mesh.coordinates()
        self.num_vertices = self.mesh.num_vertices()
        print('num of vertices',self.num_vertices)

class PeriodicBoundary(SubDomain):
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < 0.0000000000001 and x[0] > -0.000000000001 and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - 2.0
        y[1] = x[1] 

class MyFunctionSpace:
    def __init__(self, geometry, params=None):
        self.geometry = geometry
        self.params = params
    
    def generate(self, params=None):
        # self.pbc = PeriodicBoundary()
        # self.V = VectorElement("CG",self.geometry.mesh.ufl_cell(), 2,dim = 2)
        # self.P = FiniteElement( "CG",self.geometry.mesh.ufl_cell(), 2)
        # self.E = FiniteElement( "CG",self.geometry.mesh.ufl_cell(), 2)
        # self.W = FunctionSpace(self.geometry.mesh, MixedElement([self.V, self.P,self.E]),constrained_domain = self.pbc)
        self.V = VectorElement("CG",self.geometry.mesh.ufl_cell(), degree = 2,dim = 2)
        self.P = FiniteElement("CG",self.geometry.mesh.ufl_cell(), degree = 2)
        self.E = FiniteElement("CG",self.geometry.mesh.ufl_cell(), degree = 2)
        self.W = FunctionSpace(self.geometry.mesh, MixedElement([self.V, self.P,self.E]))


class MySolver:
    def __init__(self, geometry, function_space, params=None):
        self.geometry = geometry
        self.function_space = function_space
        self.params = params
        self.time = 0
        self.T = self.params['T']
        self.dt = self.params['dt']
        self.epoch = 0
        self.theta = 0.5
        self.pr = 1
        self.Ra = 1E6
        self.u_0  = Expression(('0','0','0','1.0'),degree = 2)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.R_star = (self.pr/self.Ra)**(1/2)
        self.P_star = (self.pr*self.Ra)**(-1/2)
        
    def generate_variable(self):
        self.geometry.generate()
        self.function_space.generate()
        self.V = self.function_space.V
        self.P = self.function_space.P
        self.E = self.function_space.E
        self.W = self.function_space.W

    def generate_bc(self, ctr = 0.0):
        self.bndry = self.geometry.bndry
        
        self.const = 0.0
        self.amp = 0.0
        self.ctr = ctr
        self.bc_bottom = Expression('a+b*sin(2*pi*x[0])',a = self.const,b = self.amp, pi = np.pi, degree = 2)
        
        self.noslip = Constant((0, 0))
        # velo boundary condition
        self.bcv_t = DirichletBC(self.W.sub(0), Constant((ctr, 0)), self.bndry, 3)
        self.bcv_b = DirichletBC(self.W.sub(0), Constant((0, 0)), self.bndry, 4)
        self.bcv_l = DirichletBC(self.W.sub(0), Constant((0, 0)), self.bndry, 1)
        self.bcv_r = DirichletBC(self.W.sub(0), Constant((0, 0)), self.bndry, 2)
        
        # temp boundary condition
        self.bct_t = DirichletBC(self.W.sub(2), Expression('0.0',degree = 2), self.bndry, 3)
        self.bct_b = DirichletBC(self.W.sub(2), self.bc_bottom, self.bndry, 4)
        
        # pressure boundary condition
        # self.bcp_t = DirichletBC(self.W.sub(1), Expression('0.0',degree = 2), self.bndry, 3)
        self.bcp_b = DirichletBC(self.W.sub(1), Expression('0.0',degree = 2), self.bndry, 4)
        
        self.bcs = [self.bcv_t, self.bcv_b, self.bct_t, self.bct_b]# ,self.bcp_t,self.bcp_b, self.bcv_l, self.bcv_r, 

    def generate_solver(self):
        (self.v_, self.p_, self.e_) = TestFunctions(self.W)
        self.w = Function(self.W)
        self.w.interpolate(self.u_0)
        (self.v, self.p, self.e) = split(self.w)

        self.w_old = Function(self.W)
        self.w_old.interpolate(self.u_0)
        (self.v_old, self.p_old, self.e_old) = split(self.w_old)
        def a(v,u,e) :
            D = sym(grad(v))
            return (inner(grad(v)*v, u) +self.R_star* inner(2*D, grad(u)) - inner(e,u[1]))*dx

        def b(q,v) :
            return inner(div(v),q) * dx

        def c(v,e,g) :
            return ( self.P_star*inner(grad(e),grad(g)) + inner(v,grad(e))*g )*dx

        self.F0_eq1 = a(self.v_old,self.v_,self.e_old) + b(self.p,self.v_)
        self.F0_eq2 = b(self.p_,self.v)
        self.F0_eq3 = c(self.v_old,self.e_old,self.e_)
        self.F0 = self.F0_eq1 + self.F0_eq2 #+ self.F0_eq3
        self.F0 = self.F0_eq1 + self.F0_eq2 #+ self.F0_eq3
        self.F1_eq1 = a(self.v,self.v_,self.e) + b(self.p,self.v_)
        self.F1_eq2 = b(self.p_,self.v)
        self.F1_eq3 = c(self.v,self.e,self.e_)
        self.F1 = self.F1_eq1 + self.F1_eq2 #+ self.F1_eq3
        self.F1 = self.F1_eq1 + self.F1_eq2 #+ self.F1_eq3
        # self.F = (inner((self.v-self.v_old),self.v_)/self.dt + inner((self.e-self.e_old),self.e_)/self.dt)*dx + (1.0-self.theta)*self.F0 + self.theta*self.F1
        self.F = (inner((self.v-self.v_old),self.v_)/self.dt)*dx + (1.0-self.theta)*self.F0 + self.theta*self.F1
        self.J = derivative(self.F, self.w)
        self.problem =  NonlinearVariationalProblem(self.F, self.w, self.bcs, self.J)
        self.solver = NonlinearVariationalSolver(self.problem)
        self.prm =self.solver.parameters
        #info(prm,True)  #get full info on the parameters
        self.prm['nonlinear_solver'] = 'newton'
        self.prm['newton_solver']['absolute_tolerance'] = 1E-10
        self.prm['newton_solver']['relative_tolerance'] = 1e-10
        self.prm['newton_solver']['maximum_iterations'] = 30
        self.prm['newton_solver']['linear_solver'] = 'mumps'

    def init_solve(self, ctr=0.0):
        self.generate_variable()
        self.generate_bc(ctr)
        self.generate_solver()
        self.generate_grid()
        self.time = 0 
    
    def direct_solve(self, epoch):
        for i in range(epoch):
            print('# ', i)
            self.time += self.dt
            self.epoch +=1
            self.solver.solve()
            self.w_old.assign(self.w)
            self.plot_all()

    def step_forward(self, ctr=0.0):
        self.time += self.dt
        self.epoch +=1
        self.solver.solve()
        self.w_old.assign(self.w)
        temp, velo, p ,a1 , b1= self.get_obs()
        #self.plot_all()
        return temp , velo , p , a1 , b1 

    def _get_done(self):
        return self.time > self.T  

    def generate_grid(self):
        gsx = self.params['dimx'] #30
        gsy = self.params['dimy'] #30
        xs = np.linspace(self.geometry.min_x, self.geometry.max_x, gsx)
        ys = np.linspace(self.geometry.min_y, self.geometry.max_y, gsy)
        mx, my = np.meshgrid(xs, ys)
        grids = np.stack((mx, my), 2)
        self.grids = grids
        print(f'grids: {grids.shape}')
        self.meshgrid = [mx, my]
            
    def get_obs(self):
        nu = self.params['dimy']*self.params['dimx']
        # print('nu: ', nu)
        shape = [self.params['dimy'], self.params['dimx']]
        # print('shape: ', shape)
        temp = np.array(self.w.compute_vertex_values()[3*nu:].reshape(shape))
        p = np.array(self.w.compute_vertex_values()[2*nu:3*nu].reshape(shape))
        u =  np.array(self.w.compute_vertex_values()[:1*nu].reshape(shape))
        v =  np.array(self.w.compute_vertex_values()[1*nu:2*nu].reshape(shape))
        velo = np.concatenate([np.expand_dims(u,axis = -1),np.expand_dims(v,axis = -1)],axis = -1)
        return temp, velo, p, self.ctr, self.amp
    
    def plot_all(self):
        temp, velo, p ,_ , _  = self.get_obs()
        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.meshgrid[0],self.meshgrid[1], temp, 200, cmap='jet')
        plt.colorbar()
        plt.savefig('./pics/img_temp/pic-{}.png'.format(self.epoch))

        fig = plt.figure()
        plt.axis('equal')
        plt.contourf(self.meshgrid[0],self.meshgrid[1],p, 200, cmap='jet')
        plt.colorbar()
        plt.savefig('./pics/img_pressure/pic-{}.png'.format(self.epoch))
        xl, xh  = self.geometry.min_x, self.geometry.max_x
        yl, yh = self.geometry.min_y, self.geometry.max_y
        
        fig, ax = plt.subplots(figsize=(12,9))
        ax.axis('equal')
        
        ax.set(xlim=(xl, xh), ylim=(yl, yh))
        ax.quiver(self.meshgrid[0],self.meshgrid[1], velo[:,:,0],velo[:,:,1] , temp)
        plt.savefig('./pics/img_velo/pic-{}.png'.format(self.epoch))


    def forward(self,w_tn, w_tnp1,a1,b1):
        self.generate_variable()
        
        self.generate_bc(a1=a1[0],b1 = b1[0])
        self.generate_solver(w_tn =w_tn , w_tnp1 = w_tnp1,data = 0)
        
        #self.w.vector()[:] = w_tn.reshape(-1)[dof_to_vertex_map(self.W)[:-4*(self.geometry.n+1)].astype('int32')].clone().detach().cpu().numpy()
        #self.w_old.vector()[:] = w_tn.reshape(-1)[dof_to_vertex_map(self.W)[:-4*(self.geometry.n+1)].astype('int32')].clone().detach().cpu().numpy()
        #self.w_data = Function(self.W)
        #self.w_data.vector()[:] = w_tnp1.reshape(-1)[dof_to_vertex_map(self.W)[:-4*(self.geometry.n+1)].astype('int32')].clone().detach().cpu().numpy()
        
        
        temp, velo, p ,top , bottom = self.step_forward()
        print('J = ',J)
        print('loss',np.max(np.abs(self.w_data.vector()[:] - self.w.vector()[:])))
        print('lossu',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,0] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,0])))
        print('lossv',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,1] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,1])))
        print('lossp',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,2] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,2])))
        print('losst',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,3] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,3])))
        # self.loss = (self.w - self.w_data)**2*dx
        # J = assemble(self.loss)
        # print(J)
        
        #return self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(51,151,4)[:,:,3],self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(51,151,4)[:,:,3],w_tn[:,:,3].detach().cpu().numpy(),w_tnp1[:,:,3].detach().cpu().numpy(),self.meshgrid
        return self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,2] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,2],self.meshgrid
    
    def backward(self,w_tn, w_tnp1,a1,b1):
        self.generate_variable()
        
        self.generate_bc(a1=a1[0],b1 = b1[0])
        self.generate_solver(w_tn =w_tn , w_tnp1 = w_tnp1,data = 1)
        
        temp, velo, p ,top , bottom = self.step_forward()
        self.loss = (self.w.sub(0) - self.w_data.sub(0))**2*dx+(self.w.sub(2) - self.w_data.sub(2))**2*dx+ 0.0001*(self.w.sub(1) - self.w_data.sub(1))**2*dx
        J = assemble(self.loss)
        print('J = ', J)
        grad_en, grad_enp1 = compute_gradient(J,[self.w_c,self.w_c1])
        print('loss',np.max(np.abs(self.w_data.vector()[:] - self.w.vector()[:])))
        print('lossu',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,0] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,0])))
        print('lossv',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,1] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,1])))
        print('lossp',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,2] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,2])))
        print('losst',np.max(np.abs(self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,3] - self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(33,65,4)[:,:,3])))
        # self.loss = (self.w - self.w_data)**2*dx
        # J = assemble(self.loss)
        # print(J)
        
        #return self.w.vector()[:][vertex_to_dof_map(self.W)].reshape(51,151,4)[:,:,3],self.w_data.vector()[:][vertex_to_dof_map(self.W)].reshape(51,151,4)[:,:,3],w_tn[:,:,3].detach().cpu().numpy(),w_tnp1[:,:,3].detach().cpu().numpy(),self.meshgrid
        return grad_en,grad_enp1,self.meshgrid
    def RBC_step(self,w_tn, w_tnp1,a1,b1):
    
        self.batch_size1 = w_tn.shape[0]
        self.grad_wn = torch.zeros_like(w_tn)
        self.grad_wnp1 = torch.zeros_like(w_tn)
        #self.generate_variable()
        for epoch in range(self.batch_size1):
            self.generate_bc(a1[epoch,0],b1[epoch,0])
            self.generate_solver(w_tn =w_tn , w_tnp1 = w_tnp1,data = 1)
            self.step_forward()
            self.loss = (self.w.sub(0) - self.w_data.sub(0))**2*dx+(self.w.sub(2) - self.w_data.sub(2))**2*dx+ 0.0001*(self.w.sub(1) - self.w_data.sub(1))**2*dx
            self.J1 = assemble(self.loss)
            self.grad_en, self.grad_enp1 = compute_gradient(self.J1,[self.w_c,self.w_c1])
            #grad_e = torch.Tensor(self.compute_loss().vector()[:]).to(self.device)
            
            self.grad_wn[epoch] = torch.Tensor(self.grad_en.vector()[:][vertex_to_dof_map(self.W)].reshape(w_tn.shape[1:])).to(self.device)
            self.grad_wnp1[epoch] = torch.Tensor(self.grad_enp1.vector()[:][vertex_to_dof_map(self.W)].reshape(w_tn.shape[1:])).to(self.device)
        return self.grad_wn,self.grad_wnp1,self.J1
        
        

    
    


    



        
       

   
