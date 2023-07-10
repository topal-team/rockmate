# ====================
# = global variables =
# ====================
from rkgb.utils.imports import torch, sys

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


# ==========================
#  === LISTS OF FUNCTIONS ===
# ==========================

list_python_modules = [
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
# -> ONLY used for root nodes
# -> ie nodes without depedencies

list_cheap_fct = [
    "torch.add",
    "torch.sub",
    "torch.mul",
    "torch.div",
    "torch.floor_divide",
]
# -> OPTIONAL

list_cheap_fct.extend(["list constructor", "tuple constructor"])
# because we treat them in the same way

list_view_fct = [
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

torchscript_dtype_numbers = [
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.float64,
    None, # 8
    torch.complex64,
    torch.complex128,
    torch.bool,
    None, # 12
    None, # 13
    None, # 14
    torch.bfloat16
]
default_dtype = torch.float32

def get_torchscript_dtype(t):
    if isinstance(t,torch.dtype): return t
    else:
        if not isinstance(t,int): raise Exception(
            f"TorchScript usually changes torch.dtype by "\
            f"weird integers, but here it's neither a dtype "\
            f"nor an integer, what is it ??? : {t}"
        )
        if t > 15: 
            dt = default_dtype
            problem = True
        else:
            dt = torchscript_dtype_numbers[t]
            if dt is None:
                dt = default_dtype
                problem = True
            else: problem = False
        if problem: print("Warning : "\
            f"TorchScript usually changes torch.dtype by "\
            f"weird integers. For basic dtypes we know their "\
            f"corresponding numbers, but here a {t} was found "\
            f"what is the corresponding dtype ?? \n"\
            f"{default_dtype} is used as the default dtype",
            file = sys.stderr
        )
        return dt
"""
# torchscript_dtype_numbers tabular has been created
# using the following code. jit.trace replaces dtype keywords
# by integers, which make the code impossible to run (WTF)

def test_dtype(dtype):
    class T(torch.nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x):
            return x.to("cpu",dtype=dtype)

    t = T()
    x = torch.randn(5)
    tm = torch.jit.trace_module(t,{"forward":x})
    for c in tm.code:
        try:
            n = int(c)
            print(n,end="")
        except: pass

for dtype in float_dtype + int_dtype + bool_dtype:
    print(dtype,end=" ") ; test(dtype) ; print("")
"""


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
