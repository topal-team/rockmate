import torch
import gc

def get_model_sample(name='DiT', device='cpu'):
    match name:
        case "UNet2D":
            from .unet import UNet2D
            model = UNet2D(in_channels=1, out_channels=2, block_out_channels=(32, 32, 32, 32),
                    regular_padding_mode=['replicate', 'circular'],
                    up_interpolation_mode='linear')
            sample = [torch.randn((1, 1, 32, 32), device=device)]
        case "Autoencoder2D":
            from .autoencoder import Autoencoder2D
            model = Autoencoder2D(in_channels=2, out_channels=2, kernel=[3, 5], 
                            block_out_channels=[160, 224, 352, 576, 960], 
                            regular_padding_mode=['replicate', 'circular'],
                            up_interpolation_mode='linear')
            sample = [torch.randn((1, 2, 32, 32), device=device)]
        case "DiT":
            from .dit import DiT
            model = DiT(seq_size=(45, 90),
            seq_dim=2*934,
            hidden_dim=1152,
            regular_padding_mode='constant',
            out_dim=None,
            depth=2,
            num_heads=16,
            mlp_ratio=4.0)
            sample = torch.randn((1, 2*934, 45, 90))
            sample = model.img_to_seq_pixel(sample)
            sample = (sample.to(device), torch.zeros((1, ), device=device))

    return  model.to(device), sample


