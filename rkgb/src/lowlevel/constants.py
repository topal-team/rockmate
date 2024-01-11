# ====================
# = global variables =
# ====================
import torch
import sys


init_target_string = "__sources__"
constructor_function_string = "__constructor__"
getattr_function_string = str(getattr)

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

list_random_functions = [
    "randn",
    "dropout",
    "rand",
    "randint",
    "randperm",
    "empty",
    "rrelu",
]
list_cheap_functions = [
    "add",
    "sub",
    "mul",
    "div",
    "floor_divide",
]
list_inplace_functions = [
]
list_view_functions = [
    "slice",
    "select",
    "adjoint",
    "as_strided",
    "detach",
    "diagonal",
    "expand",
    "expand_as",
    "movedim",
    "narrow",
    "permute",
    "select",
    "squeeze",
    "transpose",
    "view_as_real",
    "unflatten",
    "unfold",
    "unsqueeze",
    "view",
    "view_as",
    "unbind",
    "split",
    "hsplit",
    "vsplit",
    "tensor_split",
    "split_with_sizes",
    "swapaxes",
    "swapdims",
    "chunk",
    "values",
    "indices",
]
#  list imported from https://pytorch.org/docs/stable/tensor_view.html
# and then I tried to adapt it to 'torch.ops.aten'
for list_of_functions in [
    list_random_functions,
    list_cheap_functions,
    list_inplace_functions,
    list_view_functions
]:
    original_list = list(list_of_functions)
    for fct in original_list:
        list_of_functions.append(fct+".default")
        list_of_functions.append(fct+".Tensor")


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
