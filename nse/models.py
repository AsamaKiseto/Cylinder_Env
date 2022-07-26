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
    def __init__(self, modes1, modes2, width, last=False):
        super(FNO_layer, self).__init__()
        """ ...
        """
        self.last = last

        self.conv = SpectralConv2d(width, width, modes1, modes2)
        self.w = nn.Conv2d(width, width, 1)
        # self.bn = torch.nn.BatchNorm2d(width)

    def forward(self, x):
        """ x: (batch, hidden_channels, dim_x, dim_t)"""

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
        self.fc0 = nn.Linear(4, self.width)       # input dim: state_dim=3, control_dim=1

        self.net = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
        self.net += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.net = nn.Sequential(*self.net)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 5)
        
        # self.fc3 = nn.Linear(ny*nx*3, 2)
        # self.fcC = C_net(activate=nn.ReLU(), num_hiddens=[ny*nx*3, 1024, 512, 64, 2])

    def forward(self, x, f):
        """ 
        - x: (batch, dim_x, dim_y, dim_feature)
        """
        batch_size, ny, nx = x.shape[0], x.shape[1], x.shape[2]
        f = f.reshape(-1, 1, 1, 1).repeat(1, ny, nx, 1)
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = torch.cat((x, f), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

        x = self.net(x)

        # x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 1) # pad the domain if input is non-periodic
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        Cd = torch.mean(x[:, :, :, 3].reshape(x.shape[0], -1), 1)
        Cl = torch.mean(x[:, :, :, 4].reshape(x.shape[0], -1), 1)
        
        return x[:, :, :, :3], Cd, Cl

    def get_grid(self, shape, device):
        batchsize, ny, nx = shape[0], shape[1], shape[2]
        gridy = torch.tensor(np.linspace(0, 2.2, ny), dtype=torch.float)
        gridy = gridy.reshape(1, ny, 1, 1).repeat([batchsize, 1, nx, 1])
        gridx = torch.tensor(np.linspace(0, 1, nx), dtype=torch.float)
        gridx = gridx.reshape(1, 1, nx, 1).repeat([batchsize, ny, 1, 1])
        return torch.cat((gridy, gridx), dim=-1).to(device)      


class state_en(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_en, self).__init__()

        self.fc0 = nn.Linear(5, width)
        self.down = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
        self.down += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.down = nn.Sequential(*self.down)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)    # [batch_size, ny, nx, 5]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x_latent = self.down(x) 

        return x_latent     # [batch_size, width, ny, nx]
    
    def get_grid(self, shape, device):
        batchsize, ny, nx = shape[0], shape[1], shape[2]
        gridy = torch.tensor(np.linspace(0, 2.2, ny), dtype=torch.float)
        gridy = gridy.reshape(1, ny, 1, 1).repeat([batchsize, 1, nx, 1])
        gridx = torch.tensor(np.linspace(0, 0.41, nx), dtype=torch.float)
        gridx = gridx.reshape(1, 1, nx, 1).repeat([batchsize, ny, 1, 1])
        return torch.cat((gridy, gridx), dim=-1).to(device)


class state_de(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(state_de, self).__init__()

        self.up = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
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

        return x    # [batch_size, ny, nx, 5]


class control_en(nn.Module):
    def __init__(self, ny, nx, out_channels):
        super(control_en, self).__init__()
        self.ny, self.nx = ny, nx
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(64, 32, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(32, 16, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(16, out_channels, 5, padding=2),
        )

    def forward(self, f):
        # [batch_size] ——> [batch_size, ny, nx, 1]
        f = f.reshape(f.shape[0], 1, 1, 1).repeat(1, self.ny, self.nx, 1)   
        f = f.permute(0, 3, 1, 2)
        f = self.net(f)
        return f    # [batch_size, out_channels, ny, nx]
    

class control_de(nn.Module):
    def __init__(self, in_channels):
        super(control_de, self).__init__()
        self.in_channels = in_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(64, 32, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(32, 16, 5, padding=2),
            # nn.Tanh(),
            nn.Conv2d(16, 1, 5, padding=2),
        )

    def forward(self, f):
        f = self.net(f)
        f = torch.mean(f.reshape(f.shape[0], -1), 1)
        return f    # [batch_size]


class trans_net(nn.Module):
    def __init__(self, modes1, modes2, width, L):
        super(trans_net, self).__init__()

        self.trans = [ FNO_layer(modes1, modes2, width) for i in range(self.L-1) ]
        self.trans += [ FNO_layer(modes1, modes2, width, last=True) ]
        self.trans = nn.Sequential(*self.trans)

    def forward(self, x_latent, f_latent):
        trans_in = torch.cat((x_latent, f_latent), dim=1)
        trans_out = self.trans(trans_in)

        return trans_out


class FNO_ensemble(nn.Module):
    def __init__(self, modes1, modes2, width, L, f_channels = 4):
        super(FNO_ensemble, self).__init__()

        self.stat_en = state_en(modes1, modes2, width, L)
        self.stat_de = state_de(modes1, modes2, width, L)

        self.ctr_en = control_en(f_channels)
        self.ctr_de = control_de(f_channels)

        self.trans = trans_net(modes1, modes2, width, L)

    def forward(self, x, f):
        # x: [batch_size, ny, nx, 3]; f: [1]
                
        x_latent = self.stat_en(x)
        x_rec = self.stat_de(x_latent)

        f_latent = self.ctr_en(f)
        f_rec = self.ctr_de(f_latent)

        trans_out = self.trans(x_latent, f_latent)
        pred = self.stat_de(trans_out)

        return pred, x_rec, f_rec, trans_out 

    def get_grid(self, shape, device):
        batchsize, ny, nx = shape[0], shape[1], shape[2]
        gridy = torch.tensor(np.linspace(0, 2.2, ny), dtype=torch.float)
        gridy = gridy.reshape(1, ny, 1, 1).repeat([batchsize, 1, nx, 1])
        gridx = torch.tensor(np.linspace(0, 0.41, nx), dtype=torch.float)
        gridx = gridx.reshape(1, 1, nx, 1).repeat([batchsize, ny, 1, 1])
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