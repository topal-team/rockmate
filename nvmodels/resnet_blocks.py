import torch

from typing import Optional, Tuple, Union

from diffusers.models.activations import get_activation

from .convolution import get_convolution

class ResnetBlock2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        t_channels: Optional[int] = None,
        groups: int = 32,
        groups_out: Optional[int] = None,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        conv_shortcut_bias: bool = True,
        conv_2d_out_channels: Optional[int] = None,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel = kernel
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, 
                                        eps=eps, affine=True)

        self.conv1 = get_convolution(in_channels, out_channels, kernel, grid_type, 
                                     regular_padding_mode=regular_padding_mode)

        if t_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = torch.nn.Linear(t_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = torch.nn.Linear(t_channels, 2 * out_channels)
            else:
                raise ValueError(f"Unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, 
                                        eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_2d_out_channels = conv_2d_out_channels or out_channels
        self.conv2 = get_convolution(out_channels, conv_2d_out_channels, kernel, grid_type, 
                                     regular_padding_mode=regular_padding_mode)

        self.nonlinearity = get_activation(non_linearity)

        self.use_in_shortcut = self.in_channels != conv_2d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            shortcut_grid = 'regular'
            
            self.conv_shortcut = get_convolution(
                in_channels,
                conv_2d_out_channels,
                kernel=1,
                grid_type=shortcut_grid,
                bias=conv_shortcut_bias
            )

    def forward(self, x, t=None):
        hidden_states = x

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if t is not None:
                if not self.skip_time_act:
                    t = self.nonlinearity(t)
                t = self.time_emb_proj(t)[:, :, None, None]

        if t is not None:
            if self.time_embedding_norm == "default":
                hidden_states = hidden_states + t
                hidden_states = self.norm2(hidden_states)
            else:
                hidden_states = self.norm2(hidden_states)
                time_scale, time_shift = torch.chunk(t, 2, dim=1)
                hidden_states = hidden_states * (1 + time_scale) + time_shift
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states)

        in_data = x
        if self.conv_shortcut is not None:
            in_data = self.conv_shortcut(in_data)

        output = (in_data + hidden_states) / self.output_scale_factor

        return output