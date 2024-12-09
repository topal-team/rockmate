
import warnings
from torch import Tensor
from rkgb.lowlevel import preprocess_samples


def check_all_cuda(
        original_mod,
        example_inputs : preprocess_samples.ExampleInputs
        ):
    all_on_cuda = True
    # - inputs -
    for (key,val) in example_inputs.dict.items():
        if isinstance(val,Tensor) and not val.is_cuda:
            all_on_cuda = False
            warnings.warn(f"Module input `{key}` isn't on cuda")
    # - parameters -
    for key,val in original_mod.named_parameters():
        if isinstance(val,Tensor) and not val.is_cuda:
            all_on_cuda = False
            warnings.warn(f"original_mod parameter `{key}` isn't on cuda")
    # - global message -
    if not all_on_cuda:
        warnings.warn(
            "You asked for graphs requiring measurement of time "\
            "and memory usage.\nBut measuring memory is only relevant "\
            "on cuda: original_mod and inputs' should be on cuda.\n"\
            "You can set argument `check_device_is_cuda` "\
            "to False to skip this warning.\n"
        )


def raise_different_devices(key1,device1,key2,device2):
    raise Exception(
        f"Carelessness ! All inputs and parameters of the original_mod\n"\
        f"must share the same device. Here {key1}'s device is \n"\
        f"{device1} and {key2}'s device is {device2}."
    )


def get_device_and_check_all_same_device(
        original_mod,example_inputs : preprocess_samples.ExampleInputs
        ):
    device = None
    key_on_device = None # name of tensor/param on 'device'
    # - inputs -
    for key,val in example_inputs.dict.items():
        if isinstance(val,Tensor):
            if device is None:
                device = val.device
                key_on_device = f"input {key}"
            elif device != val.device:
                raise_different_devices(
                    f"input {key}",val.device,
                    key_on_device,device
                )
    # - parameters -
    for key,val in original_mod.named_parameters():
        if isinstance(val,Tensor):
            if device is None:
                device = val.device
                key_on_device = f"parameter {key}"
            elif device != val.device:
                raise_different_devices(
                    f"parameter {key}",val.device,
                    key_on_device,device
                )
    if device is not None: 
        return device
    else: 
        raise Exception(
            "Sorry, at least one input or parameter should be a tensor."
        )

# ==========================