from .root import *
from . import Btools
from . import Dtools
from . import Stools
from . import Ktools

def print_inputs(nn_mod):
    s = inspect.signature(nn_mod.forward)
    p = list(s.parameters.items())
    print(f"This module has {len(p)} parameters :")
    for c in p: print(c[1])

def check_inputs(nn_mod,dict_inputs):
    s = inspect.signature(nn_mod.forward)
    p = list(s.parameters.items())
    for (inp,u) in p:
        if ((u.default is inspect._empty)
            and (not (inp in dict_inputs))):
            raise Exception(
              f"input \"{inp}\" of type {u.annotation} is missing,\n"\
              f"you can use \"print_inputs\"(nn_mod) to help you.")

def make_K(nn_mod,
    dict_inputs,
    show_debug=False,
    impose_device=False,
    D_device=None,
    K_device=None):
    ref_print_debug[0] = show_debug
    check_inputs(nn_mod,dict_inputs)
    bg = Btools.make_B(nn_mod,dict_inputs,
                       impose_device=impose_device)
    dg = Dtools.B_to_D(bg,nn_mod,dict_inputs,D_device=D_device)
    sg = Stools.D_to_S(dg,keep_sequential=True)
    kg = Ktools.S_to_K(sg,nn_mod,dict_inputs,
                       K_device=K_device)
    return kg

