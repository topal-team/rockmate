
"""
@author: Zongyi Li
This file is a modified version of the Fourier Neural Operator for 1D problem such as the (time-independent) 
Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)


        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv1d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv1d(mid_channels, out_channels, 1)
        self.gelu = nn.GELU()

        self.mlp_block = nn.Sequential(
            self.mlp1,
            self.gelu,
            self.mlp2,
        )

    def forward(self, x):
        x = self.mlp_block(x)
        return x


class integral_kernel_block(nn.Module):

    def __init__(self, width, modes1,is_gelu=True):
        super(integral_kernel_block, self).__init__()
        self.is_gelu = is_gelu
        
        self.conv = SpectralConv1d(width, width, modes1)
        self.gelu = nn.GELU()
        self.mlp = MLP(width, width, width)
        self.w = nn.Conv1d(width, width, 1)

    def forward(self, x):
        
        x1 = self.conv(x)
        x1 = self.mlp(x1)
        x2 = self.w(x)
        x = x1 + x2
        if self.is_gelu:
            x = self.gelu(x)
        
        return x

class preprocess_block(nn.Module):

    def __init__(self, width):
        super(preprocess_block, self).__init__()
        self.width = width
        self.p = nn.Linear(2, self.width) # input channel_dim is 2: (u0(x), x)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        x = self.p(x)
        x = x.permute(0, 2, 1)
        
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class projection_block(nn.Module):

    def __init__(self, width):
        super(projection_block, self).__init__()
        self.width = width
        self.q = MLP(self.width, 1, self.width * 2)  # output channel_dim is 1: u1(x)
    
    def forward(self, x):
        
        x = self.q(x)
        x = x.permute(0, 2, 1)
        
        return x

class FNO1d(nn.Sequential):
    def __init__(self, modes, width,block_number=4):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width

        self.add_module('linear', preprocess_block(self.width))
        
        for i in range(0,block_number-1):
          self.add_module('kernel_layer_'+str(i), integral_kernel_block(self.width,self.modes1,is_gelu=True))
        self.add_module('kernel_layer_'+str(block_number-1),integral_kernel_block(self.width,self.modes1,is_gelu=False))

        self.add_module('projection', projection_block(self.width))