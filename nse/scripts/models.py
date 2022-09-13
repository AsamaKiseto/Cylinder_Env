import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
    
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
            
        return x


class FNO_layer_trans(nn.Module):
    def __init__(self, modes1, modes2, width, extra_channels=0, last=False):
        super(FNO_layer_trans, self).__init__()
        """ ...
        """
        self.last = last

        self.bn = nn.BatchNorm2d(width+extra_channels)
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
    def __init__(self, modes1, modes2, width, L):
        super(state_en, self).__init__()

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
        gridx = torch.tensor(np.linspace(0, 2.2, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, 0.41, ny), dtype=torch.float)
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


class control_en(nn.Module):
    def __init__(self, nx, ny, out_channels, width=5):
        super(control_en, self).__init__()
        self.nx, self.ny = nx, ny
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

    def forward(self, f):
        # [batch_size] ——> [batch_size, nx, ny, 1]
        f = f.reshape(f.shape[0], 1, 1, 1).repeat(1, self.nx, self.ny, 1)   
        f = f.permute(0, 3, 1, 2)
        f = self.net(f)
        return f    # [batch_size, out_channels, nx, ny]
    

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

    def forward(self, f):
        f = self.net(f)
        f = torch.mean(f.reshape(f.shape[0], -1), 1)
        return f    # [batch_size]


class trans_net(nn.Module):
    def __init__(self, modes1, modes2, width, L, f_channels):
        super(trans_net, self).__init__()

        self.trans = [ FNO_layer_trans(modes1, modes2, width, f_channels) ]
        self.trans += [ FNO_layer(modes1, modes2, width) for i in range(L-2) ]
        self.trans += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.trans = nn.Sequential(*self.trans)

    def forward(self, x_latent, f_latent):
        trans_in = torch.cat((x_latent, f_latent), dim=1)
        # print(f'trans_in: {trans_in.size()}')
        trans_out = self.trans(trans_in)

        return trans_out


class state_mo(nn.Module):
    def __init__(self, modes1, modes2, width, L, f_channels):
        super(state_mo, self).__init__()

        self.net = [ FNO_layer_trans(modes1, modes2, width, f_channels) ]
        self.net += [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        # self.fc0 = nn.Linear(6, width)
        self.fc1 = nn.Linear(width, 128)
        # self.fc2 = nn.Linear(128, 3)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x, f, modify):
        if modify == False:
            return 0
        
        # f = f.reshape(f.shape[0], 1, 1, 1).repeat(1, x.shape[1], x.shape[2], 1) 
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)    # [batch_size, nx, ny, 5]
        # x = torch.cat((x, f), dim=-1) 
        # x = self.fc0(x)
        # x = x.permute(0, 3, 1, 2)
        x = torch.cat((x, f), dim=1)
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x    # [batch_size, nx, ny, 5]

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2.2, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, 0.41, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class state_mo_test(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_mo_test, self).__init__()

        self.net = [ FNO_layer(modes1, modes2, width) for i in range(L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc0 = nn.Linear(5, width)
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 3)
        # self.fc2 = nn.Linear(128, 2)

    def forward(self, x, modify):
        if modify == False:
            return 0
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    # [batch_size, nx, ny, 6]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = self.net(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x    

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 2.2, nx), dtype=torch.float)
        gridx = gridx.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridy = torch.tensor(np.linspace(0, 0.41, ny), dtype=torch.float)
        gridy = gridy.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


class FNO_ensemble(nn.Module):
    def __init__(self, params):
        super(FNO_ensemble, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        nx, ny = shape[0], shape[1]

        self.stat_en = state_en(modes1, modes2, width, L)
        self.stat_de = state_de(modes1, modes2, width, L)
        self.state_mo = state_mo(modes1, modes2, width, L, f_channels)

        self.ctr_en = control_en(nx, ny, f_channels)
        self.ctr_de = control_de(f_channels)

        self.trans = trans_net(modes1, modes2, width, L, f_channels)

    def forward(self, x, f, modify=True):
        # x: [batch_size, nx, ny, 3]; f: [1]

        # print(f'x: {x.size()}')
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)
        
        # print(f'x_rec: {x_rec.size()}')

        f_latent = self.ctr_en(f)
        f_rec = self.ctr_de(f_latent)
        # print(f'f_rec: {f_rec.size()}')

        # print(f'x_latent: {x_latent.size()}, f_latent: {f_latent.size()}')
        trans_out = self.trans(x_latent, f_latent)
        mod = self.state_mo(x_latent, f_latent, modify)

        pred = self.stat_de(trans_out)
        
        return pred, x_rec, f_rec, trans_out, mod

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridy = torch.tensor(np.linspace(0, 2.2, nx), dtype=torch.float)
        gridy = gridy.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridx = torch.tensor(np.linspace(0, 1, ny), dtype=torch.float)
        gridx = gridx.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridy, gridx), dim=-1).to(device) 


class FNO_ensemble_test(nn.Module):
    def __init__(self, params):
        super(FNO_ensemble_test, self).__init__()

        modes1 = params['modes']
        modes2 = params['modes']
        width = params['width']
        L = params['L']
        shape = params['shape']
        f_channels = params['f_channels']
        nx, ny = shape[0], shape[1]

        self.stat_en = state_en(modes1, modes2, width, L)
        self.stat_de = state_de(modes1, modes2, width, L)
        self.state_mo = state_mo_test(modes1, modes2, width, L)

        self.ctr_en = control_en(nx, ny, f_channels)
        self.ctr_de = control_de(f_channels)

        self.trans = trans_net(modes1, modes2, width, L, f_channels)

    def forward(self, x, f, modify=True):
        # x: [batch_size, nx, ny, 3]; f: [1]

        # print(f'x: {x.size()}')
        x_mod = self.state_mo(x, modify)
        x = x + x_mod
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)
        
        # print(f'x_rec: {x_rec.size()}')

        f_latent = self.ctr_en(f)
        f_rec = self.ctr_de(f_latent)
        # print(f'f_rec: {f_rec.size()}')

        # print(f'x_latent: {x_latent.size()}, f_latent: {f_latent.size()}')
        trans_out = self.trans(x_latent, f_latent)

        pred = self.stat_de(trans_out)
        
        return pred, x_rec, f_rec, trans_out

    def get_grid(self, shape, device):
        batchsize, nx, ny = shape[0], shape[1], shape[2]
        gridy = torch.tensor(np.linspace(0, 2.2, nx), dtype=torch.float)
        gridy = gridy.reshape(1, nx, 1, 1).repeat([batchsize, 1, ny, 1])
        gridx = torch.tensor(np.linspace(0, 1, ny), dtype=torch.float)
        gridx = gridx.reshape(1, 1, ny, 1).repeat([batchsize, nx, 1, 1])
        return torch.cat((gridy, gridx), dim=-1).to(device) 


class policy_net_cnn(nn.Module):
    def __init__(self):
        super(policy_net_cnn, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(64, 32, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(32, 16, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(16, 1, 5, padding=2),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        f = self.net(x)
        f = torch.mean(f.reshape(f.shape[0], -1), 1)
        return f


class PIPN(nn.Module):
    def __init__(self):
        super(PIPN, self).__init__()

        self.en = nn.Sequential(
            nn.Conv1d(6, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        self.la = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            # nn.Conv1d(128, 512, 1),
            # nn.ReLU(),
            # nn.BatchNorm1d(512),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
        )
        self.de = nn.Sequential(
            nn.Conv1d(1088, 512, 1),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Conv1d(512, 256, 1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 5, 1),
        )

    def forward(self, x, f):
        bs = x.shape[0]
        nv = x.shape[1]
        f = f.reshape(bs, 1, 1).repeat(1, nv, 1)
        x = torch.cat((x, f), -1)
        x = x.permute(0, 2, 1)
        w = self.en(x)
        v = self.la(w)
        gf = v.mean(-1).reshape(bs, -1, 1).repeat(1, 1, nv)
        u = torch.cat((w, gf), 1)
        y = self.de(u)
        y = y.permute(0, 2, 1)

        return y