import torch
import math

from warnings import warn
from functools import cache
from typing import Union, Tuple

### Initialization ###

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return math.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return math.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return math.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return math.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

def regular_pad(x, padding, padding_mode):
    if padding[0] > 0:
        x = torch.nn.functional.pad(x, (0, 0, padding[0], padding[0]), mode=padding_mode[0])
    if padding[1] > 0:
        x = torch.nn.functional.pad(x, (padding[1], padding[1], 0, 0), mode=padding_mode[1])
    
    return x


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int]],
        grid_type: str = 'regular',
        bias: bool = True,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
        init_mode: str ="kaiming_normal",
        init_weight: float = 1.,
        init_bias: float = 0.,
    ):
        super().__init__()

        if isinstance(kernel, int):
            kernel = (kernel, kernel)

        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0:
            warn('Even filter size may result in an inconsistent output shape.')
        
        if isinstance(regular_padding_mode, str):
            regular_padding_mode = (regular_padding_mode, regular_padding_mode)
        
        if grid_type == 'regular':
            conv_pad = ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
            self.pad_op = lambda x : regular_pad(x, conv_pad, regular_padding_mode)
            self.post_conv = lambda x : x
        else:
            raise ValueError(f'Grid type {grid_type} is not supported')
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel

        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel[0] * kernel[1],
            fan_out=out_channels * kernel[0] * kernel[1],
        )

        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel[0], kernel[1]], 
                                                     **init_kwargs)* init_weight)

        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if bias else None
            
    def forward(self, x):
        x = self.pad_op(x)
        x = torch.nn.functional.conv2d(x, self.weight, bias=self.bias, stride=1, padding=0)
        x = self.post_conv(x)

        return x


def get_convolution(
        in_channels: int,
        out_channels: int,
        kernel: Union[int, Tuple[int]],
        grid_type: str = 'regular',
        bias: bool = True,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
        quadrature_rule: str = 'area',
        radius: float = 1.,
    ):

    if grid_type in ['regular']:
        return Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel=kernel,
                      grid_type=grid_type,
                      bias=bias,
                      regular_padding_mode=regular_padding_mode)
    else:
        raise ValueError(f'Grid type {grid_type} is not supported.')
