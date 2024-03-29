import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.utils import *
    
#===========================================================================
# 2d fourier layers
#===========================================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1    # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixt,ioxt->boxt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        # print(x.dtype)
        return x


class FNO_layer(nn.Module):
    def __init__(self, modes1, modes2, width, extra_channels=0, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        width = width + extra_channels
        self.bn = nn.BatchNorm2d(width)
        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        # x = self.bn(x)
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            # x = self.bn(x)
            
        return x


class FNO_layer_trans(nn.Module):
    def __init__(self, modes1, modes2, width, extra_channels=0, last=False):
        super(FNO_layer_trans, self).__init__()
        """ ...
        """
        self.last = last

        self.bn = nn.BatchNorm2d(width)
        self.conv = SpectralConv2d(width+extra_channels, width, modes1, modes2)
        self.w = nn.Conv2d(width+extra_channels, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

        # x = self.bn(x)
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if not self.last:
            x = F.gelu(x)
            # x = self.bn(x)
            
        return x


class FNO(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(FNO, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.L = L
        self.padding = 6
        self.fc0 = nn.Linear(6, self.width)       # input dim: state_dim=3, control_dim=1

        self.net = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 5)

        self.conv = nn.Conv2d(self.width, 2, 5, padding=2)

        # self.fc3 = nn.Linear(ny*nx*3, 2)
        # self.fcC = C_net(activate=nn.ReLU(), num_hiddens=[ny*nx*3, 1024, 512, 64, 2])

    def forward(self, x, f):
        """ 
        - x: (batch, dim_x, dim_y, dim_feature)
        """
        batch_size, nx, ny = x.shape[0], x.shape[1], x.shape[2]
        f = f.reshape(-1, 1, 1, 1).repeat(1, nx, ny, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = torch.cat((x, f), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x = self.net(x)
        # x = F.gelu(x)

        # x = x[..., :-self.padding]
        c = self.conv(x)
        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        Cd = torch.mean(x[:, :, :, 3].reshape(x.shape[0], -1), 1)
        Cl = torch.mean(x[:, :, :, 4].reshape(x.shape[0], -1), 1)
        # Cd = torch.mean(c[:, 0].reshape(c.shape[0], -1), 1)
        # Cl = torch.mean(c[:, 1].reshape(c.shape[0], -1), 1)
        
        return x[:, :, :, :3], Cd, Cl

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2.2, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, 0.41, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)      


class state_en(nn.Module):
    def __init__(self, modes1, modes2, width, L, Lx=2.2, Ly=0.41):
        super(state_en, self).__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.fc0 = nn.Linear(5, width)
        self.down = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.down += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.down = nn.Sequential(*self.down)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    # [batch_size, nx, ny, 5]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x_latent = self.down(x) 

        return x_latent     # [batch_size, width, nx, ny]
    
    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.Lx, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, self.Ly, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class state_de(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_de, self).__init__()

        self.up = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.up += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.up = nn.Sequential(*self.up)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 5)

    def forward(self, x_latent):
        x = self.up(x_latent)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x    # [batch_size, nx, ny, 5]


class state_de_rbc(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_de_rbc, self).__init__()

        self.up = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.up += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.up = nn.Sequential(*self.up)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x_latent):
        x = self.up(x_latent)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x    # [batch_size, nx, ny, 5]


class control_en(nn.Module):
    def __init__(self, out_channels, width=5):
        super(control_en, self).__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, out_channels, width, padding=2),
        )

    def forward(self, ctr):
        # [batch_size] ——> [batch_size, nx, ny, 1]
        ctr = ctr.permute(0, 3, 1, 2)
        ctr = self.net(ctr)
        return ctr    # [batch_size, out_channels, nx, ny]
    

class control_de(nn.Module):
    def __init__(self, in_channels, width=5):
        super(control_de, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(64, 32, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(32, 16, width, padding=2),
            nn.Tanh(),
            nn.Conv2d(16, 1, width, padding=2),
        )

    def forward(self, ctr):
        ctr = self.net(ctr)
        return ctr    # [batch_size]


class trans_net(nn.Module):
    def __init__(self, modes1, modes2, width, L, f_channels):
        super(trans_net, self).__init__()

        self.trans = [ FNO_layer_trans(modes1, modes2, width, f_channels) ]
        self.trans += [ FNO_layer(modes1, modes2, width) for i in range(L-2) ]
        self.trans += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.trans = nn.Sequential(*self.trans)

    def forward(self, x_latent, ctr_latent):
        trans_in = torch.cat((x_latent, ctr_latent), dim=1)
        trans_out = self.trans(trans_in)

        return trans_out


class state_mo(nn.Module):
    def __init__(self, params):
        super(state_mo, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L'] + 2
        self.Lx, self.Ly = params['Lxy']

        # self.net = [ FNO_layer_trans(modes1, modes2, width, f_channels) ]
        self.net = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc0 = nn.Linear(15, width)  # (dim_u = 2 + dim_grad_u = 4 + dim_grad_p = 2 + dim_laplace_u = 2 + dim_u_next = 2 + dim_grid = 2 + dim_f = 1 = 15)
        self.fc1 = nn.Linear(width, 128)
        # self.fc2 = nn.Linear(128, 3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, ctr, x_next):
        grid = self.get_grid(x.shape, x.device) # 2
        # ctr # 1
        u_bf = x[..., :-1]   # 2
        p_bf = x[..., -1].reshape(-1, x.shape[1], x.shape[2], 1)
        u_af = x_next[..., :-1]  # 2
        ux, uy = fdmd2D(u_bf, x.device, self.Lx, self.Ly)   # input 2 + 2
        px, py = fdmd2D(p_bf, x.device, self.Lx, self.Ly)
        uxx, _ = fdmd2D(ux, x.device, self.Lx, self.Ly)
        _, uyy = fdmd2D(uy, x.device, self.Lx, self.Ly) 
        u_lap = uxx + uyy   # input 2
        p_grad = torch.cat((px, py), -1)    # input 2
        ipt = torch.cat((grid, u_bf, ctr, u_af, ux, uy, p_grad, u_lap), -1)
        opt = self.fc0(ipt).permute(0, 3, 1, 2)
        opt = self.net(opt).permute(0, 2, 3, 1)
        opt = self.fc1(opt)
        opt = F.gelu(opt)
        opt = self.fc2(opt)

        return opt    # [batch_size, nx, ny, 5]

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.Lx, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, self.Ly, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class state_mo_prev(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_mo_prev, self).__init__()

        self.net = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x


class FNO_ensemble(nn.Module):
    def __init__(self, params):
        super(FNO_ensemble, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        Lx, Ly = params['Lxy']
        self.nx, self.ny = shape[0], shape[1]

        self.stat_en = state_en(modes1, modes2, width, L, Lx, Ly)
        self.stat_de = state_de(modes1, modes2, width, L)
        # self.state_mo = state_mo(modes1, modes2, width, L+2)

        self.ctr_en = control_en(f_channels)
        self.ctr_de = control_de(f_channels)

        self.trans = trans_net(modes1, modes2, width, L, f_channels)

    # def forward(self, x, f, modify=True):
    def forward(self, x, ctr):
        # x: [batch_size, nx, ny, 3]; f: [1]
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)

        ctr_latent = self.ctr_en(ctr)
        ctr_rec = self.ctr_de(ctr_latent)

        trans_out = self.trans(x_latent, ctr_latent)
        
        pred = self.stat_de(trans_out)
        
        return pred, x_rec, ctr_rec, trans_out #, mod


class FNO_ensemble_test(nn.Module):
    def __init__(self, params):
        super(FNO_ensemble_test, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        self.nx, self.ny = shape[0], shape[1]

        self.stat_en = state_en(modes1, modes2, width, L)
        self.stat_de = state_de(modes1, modes2, width, L)
        self.state_mo = state_mo_prev(modes1, modes2, width, L)

        self.ctr_en = control_en(f_channels)
        self.ctr_de = control_de(f_channels)

        self.trans = trans_net(modes1, modes2, width, L, f_channels)

    # def forward(self, x, f, modify=True):
    def forward(self, x, ctr):
        # x: [batch_size, nx, ny, 3]; f: [1]
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)

        ctr_latent = self.ctr_en(ctr)
        ctr_rec = self.ctr_de(ctr_latent)

        trans_out = self.trans(x_latent, ctr_latent)
        mod = self.state_mo(trans_out)
        
        pred = self.stat_de(trans_out)
        
        return pred, x_rec, ctr_rec, trans_out, mod


class FNO_ensemble_RBC(nn.Module):
    def __init__(self, params):
        super(FNO_ensemble_RBC, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        Lx, Ly = params['Lxy']
        self.nx, self.ny = shape[0], shape[1]

        self.stat_en = state_en(modes1, modes2, width, L, Lx, Ly)
        self.stat_de = state_de_rbc(modes1, modes2, width, L)
        # self.state_mo = state_mo(modes1, modes2, width, L+2)

        self.ctr_en = control_en(f_channels)
        self.ctr_de = control_de(f_channels)

        self.trans = trans_net(modes1, modes2, width, L, f_channels)

    # def forward(self, x, f, modify=True):
    def forward(self, x, ctr):
        # x: [batch_size, nx, ny, 3]; f: [1]
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)

        # ctr encode & decode
        ctr_latent = self.ctr_en(ctr)
        ctr_rec = self.ctr_de(ctr_latent)

        # trans layer
        trans_out = self.trans(x_latent, ctr_latent)
        pred = self.stat_de(trans_out)
        
        return pred, x_rec, ctr_rec, trans_out #, mod


class FNO_ensemble_RBC1(nn.Module):
    def __init__(self, params):
        super(FNO_ensemble_RBC1, self).__init__()
        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        Lx, Ly = params['Lxy']
        self.nx, self.ny = shape[0], shape[1]

        self.stat_en = state_en(modes1, modes2, width, L, Lx, Ly)
        self.stat_de = state_de_rbc(modes1, modes2, width, L)

        self.trans = trans_net(modes1, modes2, width, L, f_channels)

    # def forward(self, x, f, modify=True):
    def forward(self, x, ctr):
        # x: [batch_size, nx, ny, 3]; f: [1]
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)
        
        ctr_latent = ctr.permute(0, 3, 1, 2)
        trans_out = self.trans(x_latent, ctr_latent)
        pred = self.stat_de(trans_out)
        return pred, x_rec, ctr, trans_out
