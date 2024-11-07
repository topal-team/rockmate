import torch

from typing import Tuple, Optional, Union

from diffusers.models.activations import get_activation

from .convolution import get_convolution
from .resnet_blocks import ResnetBlock2D
from .resample import Resample2d
from .unet_blocks import UNetMidBlock2D

class DownBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        t_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_embedding_norm: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    grid_type=grid_type,
                    kernel=kernel,
                    out_channels=out_channels,
                    t_channels=t_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm=resnet_time_embedding_norm,
                    output_scale_factor=resnet_output_scale_factor,
                    regular_padding_mode=regular_padding_mode
                )
            )

        self.resnets = torch.nn.ModuleList(resnets)

        self.downsampler = None
        self.down_conv = None
        if add_downsample:
            self.downsampler = Resample2d(down=True, grid_type=grid_type, resample_filter=None, 
                                            regular_padding_mode=regular_padding_mode)
            self.explicit_down = False

            self.down_conv = get_convolution(out_channels, out_channels, kernel, grid_type, 
                                             regular_padding_mode=regular_padding_mode)

    def forward(self, x, t=None):
        for resnet in self.resnets:
            x = resnet(x, t)

        if self.down_conv is not None:
            if not self.explicit_down:
                x = self.downsampler(x)    
                x = self.down_conv(x)
            else:
                shape = list(x.shape[-2:])
                shape[0] //= 2
                shape[1] //= 2
                x = self.down_conv(x, tuple(shape))

        return x


class UpBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        t_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_embedding_norm: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
        interpolation_mode: Optional[str] = None
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    grid_type=grid_type,
                    kernel=kernel,
                    out_channels=out_channels,
                    t_channels=t_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm=resnet_time_embedding_norm,
                    output_scale_factor=resnet_output_scale_factor,
                    regular_padding_mode=regular_padding_mode,
                )
            )

        self.resnets = torch.nn.ModuleList(resnets)

        if add_upsample:
            if interpolation_mode is None:
                interpolation_mode = 'linear'

            self.upsampler = Resample2d(grid_type=grid_type, resample_filter=interpolation_mode, 
                                        regular_padding_mode=regular_padding_mode)
            
            self.up_conv = get_convolution(out_channels, out_channels, kernel, grid_type, 
                                           regular_padding_mode=regular_padding_mode)
        else:
            self.up_conv = None

    def forward(self, x, t=None, shape=None):
        for resnet in self.resnets:
            x = resnet(x, t)

        if self.up_conv is not None:
            x = self.upsampler(x, shape)
            x = self.up_conv(x)

        return x


class Encoder(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_latent_channels: bool = True,
        mid_block_add_attention=True,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
    ):
        super().__init__()

        self.conv_in = get_convolution(in_channels, block_out_channels[0], kernel, grid_type, 
                                       regular_padding_mode=regular_padding_mode)

        # Down
        self.down_blocks = torch.nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                grid_type=grid_type,
                kernel=kernel,
                num_layers=layers_per_block,
                resnet_groups=norm_num_groups,
                resnet_act_fn=act_fn,
                add_downsample=not is_final_block,
                regular_padding_mode=regular_padding_mode
            )
            
            self.down_blocks.append(down_block)

        # Mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            grid_type=grid_type,
            kernel=kernel,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            regular_padding_mode=regular_padding_mode
        )

        # Out
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=block_out_channels[-1], 
                                                num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = get_activation(act_fn)

        conv_out_channels = 2 * out_channels if double_latent_channels else out_channels
        self.conv_out = get_convolution(block_out_channels[-1], conv_out_channels, kernel, grid_type, 
                                        regular_padding_mode=regular_padding_mode)

    def forward(self, x):
        x = self.conv_in(x)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x

class Decoder(torch.nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        t_channels: Optional[int] = None,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
        interpolation_mode: Optional[str] = None
    ):
        super().__init__()

        self.conv_in = get_convolution(in_channels, block_out_channels[-1], kernel, grid_type, 
                                       regular_padding_mode=regular_padding_mode)

        # Mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            grid_type=grid_type,
            kernel=kernel,
            t_channels=t_channels,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            regular_padding_mode=regular_padding_mode
        )

        # Up
        self.up_blocks = torch.nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(reversed_block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            up_block = UpBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                grid_type=grid_type,
                kernel=kernel,
                t_channels=t_channels,
                num_layers=layers_per_block + 1,
                resnet_groups=norm_num_groups,
                resnet_act_fn=act_fn,
                add_upsample=not is_final_block,
                regular_padding_mode=regular_padding_mode,
                interpolation_mode=interpolation_mode
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Out
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=block_out_channels[0], 
                                                num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = get_activation(act_fn)
        self.conv_out = get_convolution(block_out_channels[0], out_channels, kernel, grid_type, 
                                        regular_padding_mode=regular_padding_mode)

    def forward(self, x, t=None, shapes=None):
        if shapes is None:
            shapes = (None,)*len(self.up_blocks)

        x = self.conv_in(x)

        x = self.mid_block(x, t)

        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, t, shapes[i])

        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x