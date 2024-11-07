import torch
import einops

from functools import cache
from typing import Union, Tuple, Optional

from torch.nn.functional import interpolate

from .convolution import regular_pad

### Interpolation helpers ###

def constant_spline_2d(x, shape):
    x = interpolate(x, size=shape, mode='nearest-exact')

    return x

def linear_spline_2d(x, shape):
    x = interpolate(x, size=shape, mode='bilinear',
                    align_corners=False, antialias=True)
    
    return x

def cubic_spline_2d(x, shape):
    x = interpolate(x, size=shape, mode='bicubic',
                    align_corners=False, antialias=True)
    
    return x

def upsample(x, resample_filter):
    x = torch.nn.functional.conv_transpose2d(x, 
                                         resample_filter,
                                         groups=resample_filter.shape[0],
                                         stride=2, 
                                         padding=0
                                    )
    return x

def downsample(x, resample_filter):
    x = torch.nn.functional.conv2d(x, 
                                resample_filter,
                                groups=resample_filter.shape[0],
                                stride=2, 
                                padding=0
                            )
    return x

### Resample ###

class Resample2d(torch.nn.Module):
    def __init__(
        self,
        up: bool = False,
        down: bool = False,
        grid_type: str = 'regular',
        resample_filter: Optional[Union[str, list]] = 'linear',
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
    ):
        super().__init__()

        assert not (up and down)

        self.up = up
        self.down = down
        
        if isinstance(regular_padding_mode, str):
            regular_padding_mode = (regular_padding_mode, regular_padding_mode)
        

        if isinstance(resample_filter, str):
            if resample_filter == 'constant':
                self.interp = lambda x, s: constant_spline_2d(x, s)
            elif resample_filter == 'linear':
                self.interp = lambda x, s: linear_spline_2d(x, s)
            elif resample_filter == 'cubic':
                self.interp = lambda x, s: cubic_spline_2d(x, s)
            else:
                raise ValueError(f'Interpolation mode {resample_filter} is not supported.')

            self.pre_interp = lambda x : x 
            self.post_interp = lambda x : x
            
            self.register_buffer('resample_filter', None)
        else:
            assert up or down, 'Explicit resample_filter requires up or down. None=[1,1].'

            if resample_filter is None:
                resample_filter = [1,1]
            
            resample_filter = torch.as_tensor(resample_filter, dtype=torch.float32)
            resample_filter = resample_filter.ger(resample_filter) / resample_filter.sum().square()
            resample_filter = resample_filter.unsqueeze(0).unsqueeze(1)

            self.register_buffer('resample_filter', resample_filter)

            n_pad = (resample_filter.shape[-1] - 1) // 2
            resample_pad = (n_pad, n_pad)
            self.pre_interp = lambda x : regular_pad(x, resample_pad, regular_padding_mode)
            self.post_interp = lambda x : x
            
            if up:
                self.interp = lambda x: upsample(x, self.up_expand_filter(x.shape[1], x.device))
            
            if down:
                self.interp = lambda x: downsample(x, self.down_expand_filter(x.shape[1], x.device))
    
    @cache
    def up_expand_filter(self, channels, device):
        resample_filter = self.resample_filter.to(device)
        return resample_filter.mul(4).tile([channels, 1, 1, 1])
    
    @cache
    def down_expand_filter(self, channels, device):
        resample_filter = self.resample_filter.to(device)
        return resample_filter.tile([channels, 1, 1, 1])

    def forward(self, x, shape=None):
        x = self.pre_interp(x)

        if self.up or self.down:
            if self.resample_filter is not None:
                x = self.interp(x)
            else:
                if self.up:
                    shape = (2 * x.shape[-2], 2 * x.shape[-1])
                else:
                    shape = (x.shape[-2] // 2, x.shape[-1] // 2)
                
                x = self.interp(x, shape)
        else:
            if shape is not None:
                x = self.interp(x, shape)
        
        x = self.post_interp(x)

        return x