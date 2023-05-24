__all__ = [
    "get_GPT",
    "get_UNet",
    "get_MLP",
    "get_RegNet32",
    "get_ResNet101",
    "get_nn_Transformer",
    "get_FNO1d",
    "get_FNO3d",
    "get_UFNO",
    "get_UNO",
    "LossLayer",
    "get_iterator_over_all_examples",
    "sanity_check_forward_and_backward",
    "sanity_check_forward_and_backward_all_examples",
]

import torch

"""
    all 'get_model' functions return a module and a sample
    /!\ /!\ sample is a list -> for compatibility in case there are several inputs /!\ /!\ 
"""

# =========================================================

def get_GPT(device,nlayers=12,dropout=0.1,batchsize=12000):
    from .GPT import GPT2
    model = GPT2(nlayers=nlayers,dropout=dropout)
    model.to(device)
    sample = [torch.randint(batchsize,(100,20),device=device)]
    return model,sample
def get_fst_param_GPT(model):
    return model.h[0][1].layer2.c_fc.weight

# =========================================================

def get_UNet(device,batchsize=50,image_size=256):
    from .unet import UNet
    model = UNet(in_channels=3, out_channels=1, init_features=32)
    model.to(device)
    sample = [torch.randn(batchsize,3,image_size,image_size,device=device).requires_grad_()]
    return model,sample
def get_fst_param_UNet(model):
    return model.encoder1[0].weight
    
# =========================================================

def get_MLP(device,batchsize=100,image_size=256):
    from mlp_mixer_pytorch import MLPMixer
    model = MLPMixer(
        image_size = image_size,
        channels = 3,
        patch_size = 16,
        dim = 512,
        depth = 12,
        num_classes = 1000
    )
    model.to(device)
    sample = [torch.randn(batchsize, 3, image_size, image_size,device=device)]
    return model,sample
def get_fst_param_MLP(model):
    return model[1].weight
    
# =========================================================

def get_RegNet32(device,batchsize=23,image_size=256):
    from torchvision.models import regnet_x_32gf
    model = regnet_x_32gf()
    model.to(device)
    sample = [torch.randn(batchsize, 3, image_size, image_size,device=device)]
    return model,sample
def get_fst_param_RegNet32(model):
    return model.stem[0].weight
    
# =========================================================

def get_ResNet101(device,batchsize=64,image_size=256):
    from torchvision.models import resnet101
    model = resnet101()
    model.to(device)
    sample = [torch.randn(batchsize, 3, image_size, image_size,device=device)]
    return model,sample
def get_fst_param_ResNet101(model):
    return model.conv1.weight
    
# =========================================================

def get_nn_Transformer(device,batchsize=32,num_encoder_layers=6,num_decoder_layers=6):
    model = torch.nn.Transformer(
        nhead=16, 
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )
    model.to(device)

    sample = [
        torch.rand((200, batchsize, 512),device=device),
        torch.rand((200, batchsize, 512),device=device),
    ]
    return model,sample
def get_fst_param_nn_Transformer(model):
    return model.encoder.layers[0].linear1.weight
    
# =========================================================

def get_FNO1d(device,batchsize=4400,block_number=4,image_size=256):
    from models.FNO1d import FNO1d
    model = FNO1d(16, 64 ,block_number=block_number)
    model.to(device)
    sample = [torch.rand((batchsize,image_size,1),device=device)]
    return model,sample
def get_fst_param_FNO1d(model):
    return model.linear.p.weight
    
# =========================================================

def get_FNO3d(device,batchsize=5,block_number=4,size_x=64,size_y=64,size_z=40):
    from models.FNO3d import FNO3d
    model = FNO3d(16, 16, 16, 64 ,block_number=block_number)
    model.to(device)
    sample = [torch.rand((batchsize,size_x,size_y,size_z,10),device=device)]
    return model,sample
def get_fst_param_FNO3d(model):
    return model.linear.p.weight

# =========================================================

def get_UFNO(device,batchsize=3,size_x=96,size_y=200,size_z=24):
    from models.UFNO import Net3d
    mode1 = 10
    mode2 = 10
    mode3 = 10
    width = 36
    model = Net3d(mode1, mode2, mode3, width)
    model.to(device)
    sample = [torch.rand((batchsize,size_x,size_y,size_z,12),device=device)]
    return model,sample
def get_fst_param_UFNO(model):
    return model.conv1.fc0.weight
# =========================================================

def get_UNO(device,batchsize=2):
    from models.UNO import Uno3D_T40
    model = Uno3D_T40(in_width = 6, width = 8)
    model.to(device)
    S, T_in = 64, 10 
    sample = [torch.rand((batchsize, S, S, T_in, 1),device=device)]
    return model,sample
def get_fst_param_UNO(model):
    return model.fc.weight

# =========================================================



# =========================================================

class LossLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.sum()

# =========================================================



# =========================================================

dict_all_examples = dict(
    GPT=(get_GPT,get_fst_param_GPT),
    UNet=(get_UNet,get_fst_param_UNet),
    MLP=(get_MLP,get_fst_param_MLP),
    RegNet32=(get_RegNet32,get_fst_param_RegNet32),
    ResNet101=(get_ResNet101,get_fst_param_ResNet101),
    nn_Transformer=(get_nn_Transformer,get_fst_param_nn_Transformer),
    FNO1d=(get_FNO1d,get_fst_param_FNO1d),
    FNO3d=(get_FNO3d,get_fst_param_FNO3d),
    UFNO=(get_UFNO,get_fst_param_UFNO),
    UNO=(get_UNO,get_fst_param_UNO)    
)

import gc

def get_iterator_over_all_examples(device,skip_error=True,
        examples=[
            "GPT","UNet","MLP","RegNet32","ResNet101",
            "nn_Transformer","FNO1d","FNO3d","UFNO","UNO"]):
    for name in examples:
        get_fct,get_param_fct = dict_all_examples[name]
        model,sample = None,[] # To do not accumulate memory
        gc.collect()
        if skip_error:
            try:
                model,sample = get_fct(device)
            except:
                print(
                    f"/!\\ Error when trying to import {name} /!\\ \n"\
                    f"-> skip it since keyword arg skip_error=True")
            else:
                print(f"Successfully loaded {name}")
                yield name,model,sample,get_param_fct
        else:
            model,sample = get_fct(device)
            print(f"Successfully loaded {name}")
            yield name,model,sample,get_param_fct

            
def sanity_check_forward_and_backward(model,rematMod,sample,get_param_fct):
    # old 1
    model.zero_grad()
    torch.random.manual_seed(0)
    y_old = model(*sample)
    loss = y_old.sum() ; loss.backward()
    save_grad_old = get_param_fct(model).grad.view(-1)
    # new 1
    rematMod.zero_grad()
    torch.random.manual_seed(0)
    y_new = rematMod(*sample)
    print(y_new.shape)
    loss = y_new.sum() ; loss.backward()
    save_grad_new = get_param_fct(rematMod.original_mod).grad.view(-1)
    # old 2
    model.zero_grad()
    torch.random.manual_seed(0)
    y_old2 = model(*sample)
    loss = y_old2.sum() ; loss.backward()
    save_grad_old2 = get_param_fct(model).grad.view(-1)
    print("Abs sum of differences between two executions of the original module :")
    print("output :",abs(y_old-y_old2).sum())
    print("param.grad :",abs(save_grad_old - save_grad_old2).sum())
    
    print("Abs sum of differences between the original module and the remat one :")
    print("output :",abs(y_old-y_new).sum())
    print("param.grad :",abs(save_grad_old - save_grad_new).sum())
    
    print("Total abs sum of the original module results :")
    print("total output :",abs(y_old).sum())
    print("total grad :",abs(save_grad_old).sum())
            
        
def sanity_check_forward_and_backward_all_examples(device,remat_fct,skip_error=True,
        examples=[
            "GPT","UNet","MLP","RegNet32","ResNet101",
            "nn_Transformer","FNO1d","FNO3d","UFNO","UNO"]):
    # remat_fct = rockmate.main.HRemat
    print(examples)
    iterator_over_all_examples = get_iterator_over_all_examples(device,skip_error,examples)
    while True:
        model,sample = None,[] # To do not accumulate memory
        try:
            name,model,sample,get_param_fct = next(iterator_over_all_examples)
            print(f"== Start to solve {name} ==")
            rematMod = remat_fct(model,sample,budget=2e10)
            print(f"== Start sanity check of forward and backward passes of {name} ==")
            sanity_check_forward_and_backward(model,rematMod,sample,get_param_fct)
            print("\n"*2)
        except StopIteration:
            print("=== End ===")
            return ()