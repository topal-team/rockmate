import sys
import warnings
import torch
import ast
from rkgb.lowlevel import ast_add_on

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

def get_torchscript_dtype(t):
    default_dtype = torch.get_default_dtype()
    if isinstance(t,torch.dtype):
        return t
    else:
        if not isinstance(t,int): raise Exception(
            f"TorchScript usually changes torch.dtype by "\
            f"weird integers, but here it's neither a dtype "\
            f"nor an integer, what is it ??? : {t}"
        )
        if t > 15: 
            dtype = default_dtype
            unknown_so_default = True
        else:
            dtype = torchscript_dtype_numbers[t]
            if dtype is None:
                dtype = default_dtype
                unknown_so_default = True
            else: unknown_so_default = False
        if unknown_so_default: warnings.warn(
            f"TorchScript usually changes torch.dtype by "\
            f"weird integers. For basic dtypes we know their "\
            f"corresponding numbers, but here a {t} was found "\
            f"what is the corresponding dtype ?? \n"\
            f"{default_dtype} is used as the default dtype."
        )
        return dtype
    

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


def try_to_fix_dtype_in_returned_ast_code(code_ast,our_global,tmp_local):
    """
    jit replaces some default calling arguments (kwarg)
    by numbers, so in raw.py we already removed some 
    default values; but here we try to fix dtypes args
    So we look for args which int value, and check if by
    changing them to dtypes (via the correspondence table 
    we found, see functions above) it solves the problem.
    We assume only one arg refers to dtype, so we 
    try one by one and undo changes if the problem isn't
    solved. 
    Note: this function is inplace: it fixes the code if possible.
    """
    pieces_of_code_to_check = [code_ast]
    fixed = False
    while not fixed and pieces_of_code_to_check != []:
        piece_of_code = pieces_of_code_to_check.pop()
        if isinstance(piece_of_code,ast.Call):
            for i,arg in enumerate(piece_of_code.args):
                if (ast_add_on.is_constant(arg)
                and isinstance(arg.value,int)):
                    save_value = arg.value
                    piece_of_code.args[i] = (
                        ast_add_on.make_ast_constant(
                        get_torchscript_dtype(arg.value)
                    ) ) # -> inplace change
                    try:
                        exec(code_ast, our_global, tmp_local)
                        fixed[0] = True
                        break
                    except:
                        piece_of_code.args[i] = save_value
                else: pieces_of_code_to_check.append(arg)
    if not fixed: raise Exception(
        f"Sorry there are some hallucinations/bugs in the code generated"\
        f"by jit tracer which make it impossible to exec, the code "\
        f"is : {ast_add_on.ast_to_str(code_ast)}"
    )