import torch

from typing import Tuple, Optional, Union

from diffusers.models.attention_processor import Attention

from .convolution import get_convolution
from .resnet_blocks import ResnetBlock2D
from .resample import Resample2d

class DownBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        out_channels: Optional[int] = None,
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
        out_channels = in_channels if out_channels == None else out_channels

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
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
            if grid_type in ['regular']:
                self.downsampler = Resample2d(down=True, grid_type=grid_type, resample_filter=None, 
                                              regular_padding_mode=regular_padding_mode)
                self.explicit_down = False
            else:
                self.explicit_down = True

            self.down_conv = get_convolution(out_channels, out_channels, kernel, grid_type, 
                                             regular_padding_mode=regular_padding_mode)

    def forward(self, x, t=None):
        output_states = ()

        for resnet in self.resnets:
            x = resnet(x, t)
            output_states = output_states + (x,)

        if self.down_conv is not None:
            if not self.explicit_down:
                x = self.downsampler(x)    
                x = self.down_conv(x)
            else:
                shape = list(x.shape[-2:])
                shape[0] //= 2
                shape[1] //= 2
                x = self.down_conv(x, tuple(shape))

            output_states = output_states + (x,)

        return x, output_states


class UpBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        prev_output_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
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
        out_channels = in_channels if out_channels == None else out_channels
        prev_output_channels = in_channels if prev_output_channels == None else prev_output_channels

        resnets = []
        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlock2D(
                    in_channels=resnet_in_channels + res_skip_channels,
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

    def forward(self, x, x_prev, t=None, shape=None):
        for resnet in self.resnets:
            res_hidden_states = x_prev[-1]
            x_prev = x_prev[:-1]

            x = torch.cat([x, res_hidden_states], dim=1)
            x = resnet(x, t)

        if self.up_conv is not None:
            x = self.upsampler(x, shape)
            x = self.up_conv(x)

        return x


class UNetMidBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        t_channels: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_output_scale_factor: float = 1.0,
        add_attention: bool = True,
        attn_groups: Optional[int] = None,
        attention_head_dim: Optional[int] = 1,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant'
    ):
        super().__init__()
        self.resnet_in = ResnetBlock2D(
                            in_channels=in_channels,
                            grid_type=grid_type,
                            kernel=kernel,
                            out_channels=in_channels,
                            t_channels=t_channels,
                            dropout=dropout,
                            groups=resnet_groups,
                            eps=resnet_eps,
                            non_linearity=resnet_act_fn,
                            output_scale_factor=resnet_output_scale_factor,
                            regular_padding_mode=regular_padding_mode,
                        )

        attention_head_dim = in_channels if attention_head_dim is None else attention_head_dim
        attn_groups = resnet_groups if attn_groups is None else attn_groups
        
        resnets = []
        attentions = []
        for _ in range(num_layers):
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    grid_type=grid_type,
                    kernel=kernel,
                    out_channels=in_channels,
                    t_channels=t_channels,
                    dropout=dropout,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=resnet_output_scale_factor,
                    regular_padding_mode=regular_padding_mode,
                )
            )
            if add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=resnet_output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)
         
        if add_attention:
            self.pre_attn =  lambda x : x
            self.post_attn =  lambda x : x

        self.resnets = torch.nn.ModuleList(resnets)
        self.attentions = torch.nn.ModuleList(attentions)


    def forward(self, x, t=None):
        x = self.resnet_in(x, t)

        for attn, resnet in zip(self.attentions, self.resnets):
            if attn is not None:
                x = self.pre_attn(x)
                x = attn(x, temb=t)
                x = self.post_attn(x)

            x = resnet(x, t)

        return x