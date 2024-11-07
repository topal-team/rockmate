import torch

from typing import Optional, Tuple, Union

from diffusers.models.activations import get_activation
from diffusers.models.embeddings import TimestepEmbedding

from .convolution import get_convolution
from .embedding import vector_fourier_embedding
from .unet_blocks import DownBlock2D, UpBlock2D, UNetMidBlock2D


class UNet2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        block_out_channels: Tuple[int] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        dropout: float = 0.0,
        mid_block_scale_factor: float = 1,
        add_time_embedding: bool = True,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        embed_mult: int = 4,
        resnet_time_embedding_norm: str = "default",
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        add_attention: bool = True,
        attention_head_dim: Optional[int] = 8,
        attn_norm_num_groups: Optional[int] = None,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
        up_interpolation_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        # Time
        if add_time_embedding:
            self.flip_sin_to_cos = flip_sin_to_cos
            self.freq_shift = freq_shift

            self.time_project_dim = block_out_channels[0]
            self.time_embed_dim = block_out_channels[0] * embed_mult

            self.time_embedding = TimestepEmbedding(self.time_project_dim, self.time_embed_dim)
        else:
            self.time_embed_dim = None
            self.time_embedding = None

        # Input
        self.conv_in = get_convolution(in_channels, block_out_channels[0], kernel, grid_type, 
                                       regular_padding_mode=regular_padding_mode)

        # Down
        self.down_blocks = torch.nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i in range(num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = (i == num_blocks - 1)

            down_block = DownBlock2D(
                            in_channels=input_channel,
                            grid_type=grid_type,
                            kernel=kernel,
                            out_channels=output_channel,
                            t_channels=self.time_embed_dim,
                            dropout=dropout,
                            num_layers=layers_per_block,
                            resnet_eps=norm_eps,
                            resnet_time_embedding_norm=resnet_time_embedding_norm,
                            resnet_act_fn=act_fn,
                            resnet_groups=norm_num_groups,
                            add_downsample=not is_final_block,
                            regular_padding_mode=regular_padding_mode
                        )
                
            self.down_blocks.append(down_block)

        # Mid
        self.mid_block = UNetMidBlock2D(
                            in_channels=block_out_channels[-1],
                            grid_type=grid_type,
                            kernel=kernel,
                            t_channels=self.time_embed_dim,
                            dropout=dropout,
                            resnet_eps=norm_eps,
                            resnet_act_fn=act_fn,
                            resnet_output_scale_factor=mid_block_scale_factor,
                            attention_head_dim=attention_head_dim if attention_head_dim is not None else block_out_channels[-1],
                            resnet_groups=norm_num_groups,
                            attn_groups=attn_norm_num_groups,
                            add_attention=add_attention,
                            regular_padding_mode=regular_padding_mode
                        )
        # Up
        self.up_blocks = torch.nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(num_blocks):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, num_blocks - 1)]
            is_final_block = i == num_blocks - 1

            up_block = UpBlock2D(
                        in_channels=input_channel,
                        grid_type=grid_type,
                        kernel=kernel,
                        prev_output_channels=prev_output_channel,
                        out_channels=output_channel,
                        t_channels=self.time_embed_dim,
                        dropout=dropout,
                        num_layers=layers_per_block + 1,
                        resnet_eps=norm_eps,
                        resnet_time_embedding_norm=resnet_time_embedding_norm,
                        resnet_act_fn=act_fn,
                        resnet_groups=norm_num_groups,
                        add_upsample=not is_final_block,
                        regular_padding_mode=regular_padding_mode,
                        interpolation_mode=up_interpolation_mode
                    )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # Out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = torch.nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = get_activation(act_fn)
        self.conv_out = get_convolution(block_out_channels[0], out_channels, kernel, grid_type,
                                        regular_padding_mode=regular_padding_mode)

    def forward(self, x, t=None, **kwargs):

        # Embedding
        if t is not None and self.time_embedding is not None:
            time_project = vector_fourier_embedding(t, self.time_project_dim, 
                                                    self.flip_sin_to_cos, self.freq_shift)
            emb = self.time_embedding(time_project)
        else:
            emb = None

        x = self.conv_in(x)

        # Down
        down_block_res_samples = (x,)
        down_shapes = (x.data.shape,)
        for downsample_block in self.down_blocks:
            x, res_samples = downsample_block(x, emb)

            down_block_res_samples += res_samples
            down_shapes += (x.data.shape,)
        
        down_shapes = tuple(reversed(down_shapes[0:-2]))

        #  Mid
        x = self.mid_block(x, emb)

        #  Up
        shape_counter = 0
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if upsample_block.up_conv is not None:
                x = upsample_block(x, res_samples, emb, tuple(down_shapes[shape_counter][-2:]))
                shape_counter += 1
            else:
                x = upsample_block(x, res_samples, emb)
 
        # Out
        x = self.conv_norm_out(x)
        x = self.conv_act(x)
        x = self.conv_out(x)

        return x