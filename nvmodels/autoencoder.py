import torch

from typing import Optional, Tuple, Union

from diffusers.models.embeddings import TimestepEmbedding

from .convolution import get_convolution
from .embedding import vector_fourier_embedding
from .autoencoder_blocks import Encoder, Decoder



class Autoencoder2D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        grid_type: str = 'regular',
        kernel: Union[int, Tuple[int]] = (3,3),
        block_out_channels: Tuple[int] = (64, 128, 256),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        mid_block_add_attention: bool = True,
        add_time_embedding: bool = False,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        embed_mult: int = 4,
        regular_padding_mode: Union[str, Tuple[str]] = 'constant',
        up_interpolation_mode: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        self.num_blocks = len(block_out_channels)

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

        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            grid_type=grid_type,
            kernel=kernel,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_latent_channels=True,
            mid_block_add_attention=mid_block_add_attention,
            regular_padding_mode=regular_padding_mode,
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            grid_type=grid_type,
            kernel=kernel,
            t_channels=self.time_embed_dim,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            mid_block_add_attention=mid_block_add_attention,
            regular_padding_mode=regular_padding_mode,
            interpolation_mode=up_interpolation_mode
        )

        #Scaling convs
        self.encode_conv = get_convolution(2 * latent_channels, 2 * latent_channels, kernel=1,
                                           grid_type=grid_type, regular_padding_mode=regular_padding_mode)
        
        self.decode_conv = get_convolution(latent_channels, latent_channels, kernel=1,
                                           grid_type=grid_type, regular_padding_mode=regular_padding_mode)
        
    def encode(self, x):
        x = self.encoder(x)
        x = self.encode_conv(x)

        mean, logvar = torch.chunk(x, chunks=2, dim=1)

        return mean, logvar
    
    def decode(self, x, t=None, shape=None):
        # Time embedding
        if t is not None and self.time_embedding is not None:
            time_project = vector_fourier_embedding(t, self.time_project_dim, 
                                                    self.flip_sin_to_cos, self.freq_shift)
            emb = self.time_embedding(time_project)
        else:
            emb = None

        # Compute up shapes
        x_shape = tuple(x.shape[-2:])

        up_shapes = [(x_shape[0] * 2**(j+1), x_shape[1] * 2**(j+1)) for j in range(self.num_blocks - 1)]
        if shape is not None:
            up_shapes[-1] = shape
        
        up_shapes = tuple(up_shapes + [None])

        # Decode
        x = self.decode_conv(x)
        x = self.decoder(x, t=emb, shapes=up_shapes)

        return x

    def forward(self, x, t=None, return_latent=True, **kwargs):
        shape = tuple(x.shape[-2:])

        mean, logvar = self.encode(x)
        out = self.decode(mean, t=t, shape=shape)

        if return_latent:
            return out, mean, logvar
        
        return out