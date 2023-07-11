
import warnings
from torch import Tensor
from lowlevel import preprocess_samples


def check_all_cuda(
        model,dict_inputs : preprocess_samples.DictInputs
        ):
    all_on_cuda = True
    # - inputs -
    for (key,val) in dict_inputs.dict.items():
        if isinstance(val,Tensor) and not val.is_cuda:
            all_on_cuda = False
            warnings.warn(f"Module input `{key}` isn't on cuda")
    # - parameters -
    for key,val in model.named_parameters():
        if isinstance(val,Tensor) and not val.is_cuda:
            all_on_cuda = False
            warnings.warn(f"Model parameter `{key}` isn't on cuda")
    # - global message -
    if not all_on_cuda:
        warnings.warn(
            "You asked for graphs requiring measurement of time "\
            "and memory usage.\nBut measuring memory is only relevant "\
            "on cuda: model and inputs' should be on cuda.\n"\
            "You can set argument `check_device_is_cuda` "\
            "to False to skip this warning.\n"
        )


def raise_different_devices(key1,device1,key2,device2):
    raise Exception(
        f"Carelessness ! All inputs and parameters of the model\n"\
        f"must share the same device. Here {key1}'s device is \n"\
        f"{device1} and {key2}'s device is {device2}."
    )


def get_device_and_check_all_same_device(
        model,dict_inputs : preprocess_samples.DictInputs
        ):
    device = None
    k = None
    if not isinstance(dict_inputs,dict):
        dict_inputs = dict(enumerate(dict_inputs))

    for (key,inp) in dict_inputs.items():
        if isinstance(inp,Tensor):
            if device is None: device = inp.device ; k = f"input {key}"
            else:
                if device != inp.device:
                    raise_different_devices(f"input {key}",inp.device,k,device)
    i = -1
    for p in model.parameters():
        i += 1
        if d is None: d = p.device ; k = f"{i}-th parameter"
        else:
            if d != p.device:
                raise_different_devices(f"{i}-th parameter",p.device,k,device)
    if device is not None: return device
    else: raise Exception(
        "Sorry, at least one input or one parameter should be a tensor.")

# ==========================