import sys
import torch

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
    if isinstance(t,torch.dtype):
        return t
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
    

""" TO KEEP
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

