# ====================
# = global variables =
# ====================
import torch
import sys


init_target_string = "__sources__"
constructor_function_string = "__constructor__"

render_color_special = "green"

time_min_duration = 0
time_min_repeat = 5

# -> print debug messages
ref_verbose = [False]
def print_debug(*args, **kwargs):
    if ref_verbose[0]:
        print(*args, **kwargs)


# -> acceptance rate for two time measures to be declared equal
ref_reasonable_rate = [0.4]
def change_reasonable_rate(x):
    assert 0 <= x
    ref_reasonable_rate[0] = x


# -> to test phantoms detection
ref_test_phantoms_detection = [False]

# =========================
# === Custom exceptions ===
# =========================

class ExceptionModuleDoesNotReqGrad(Exception):
    # Either none of the outputs require a grad 
    # or the module returns only constants
    pass

# ==========================
# === LISTS OF FUNCTIONS ===
# ==========================

list_pytorch_packages = [
    "torch",
    "torch.nn.functional",
    "torch.Tensor",
    "torch._C._nn",
    "torch._C._fft",
    "torch.ops.aten",
]

list_rand_fct = [
    "torch.randn",
    "torch.dropout",
    "torch.rand",
    "torch.randint",
    "torch.randperm",
    "torch.empty",
    "torch.rrelu",
]
# -> ONLY used nodes without dependencies

list_cheap_functions = [
    "torch.add",
    "torch.sub",
    "torch.mul",
    "torch.div",
    "torch.floor_divide",
]

list_inplace_fct = [
]
list_view_fct = [
    "torch.ops.aten.slice",
    "torch.ops.aten.select",
    "torch.adjoint",
    "torch.Tensor.adjoint",
    "torch.as_strided",
    "torch.Tensor.as_strided",
    "torch.Tensor.detach",
    "torch.diagonal",
    "torch.Tensor.diagonal",
    "torch.Tensor.expand",
    "torch.Tensor.expand_as",
    "torch.movedim",
    "torch.Tensor.movedim",
    "torch.narrow",
    "torch.Tensor.narrow",
    "torch.permute",
    "torch.Tensor.permute",
    "torch.select",
    "torch.Tensor.select",
    "torch.squeeze",
    "torch.Tensor.squeeze",
    "torch.transpose",
    "torch.Tensor.transpose",
    "torch.view_as_real",
    "torch.Tensor.unflatten",
    "torch.Tensor.unfold",
    "torch.unsqueeze",
    "torch.Tensor.unsqueeze",
    "torch.Tensor.view",
    "torch.Tensor.view_as",
    "torch.unbind",
    "torch.Tensor.unbind",
    "torch.split",
    "torch.Tensor.split",
    "torch.hsplit",
    "torch.Tensor.hsplit",
    "torch.vsplit",
    "torch.Tensor.vsplit",
    "torch.tensor_split",
    "torch.Tensor.tensor_split",
    "torch.split_with_sizes",
    "torch.Tensor.split_with_sizes",
    "torch.swapaxes",
    "torch.Tensor.swapaxes",
    "torch.swapdims",
    "torch.Tensor.swapdims",
    "torch.chunk",
    "torch.Tensor.chunk",
    "torch.Tensor.values",
    "torch.Tensor.indices",
]
#  list imported from https://pytorch.org/docs/stable/tensor_view.html

list_batch_fct = [
    torch.nn.BatchNorm1d,
    torch.nn.BatchNorm2d,
    torch.nn.BatchNorm3d,
    torch.nn.SyncBatchNorm,
    torch.nn.InstanceNorm1d,
    torch.nn.InstanceNorm2d,
    torch.nn.InstanceNorm3d,
]

float_dtype = [
    torch.float32,
    torch.float,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16
]
int_dtype = [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64]
bool_dtype = [torch.bool]

default_forced_kwargs = dict(
    [
        (("torch.batch_norm", [("momentum", 6, 0)])),
        (("torch.instance_norm", [("momentum", 6, 0)])),
    ]
)
# This dict is used by default when force_special_kwargs=True
# -> dict of : fct_name -> (arg_name,arg_value) list to inforce
# We change some kwargs in the code to avoid changing values
# due to recomputation. For instance batchnorm statistics.
# ==========================
