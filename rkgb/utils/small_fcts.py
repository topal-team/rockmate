# ==========================
# ==== Useful functions ====
# ==========================
from rkgb.utils.imports import *
import inspect

# -> to raise exceptions with lambda functions
def raise_(s):
    raise Exception(s)

# -> to get all the attrs except special ones
def vdir(c):
    return [s for s in dir(c)
            if (not s.startswith("__")
            and not inspect.ismethod(getattr(c,s)))]

# -> strings
def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text
def remove_suffix(text, suffix):
    if text.endswith(suffix):
        return text[:-len(suffix)]
    return text

# -> for unique_num of each node
def copy_generator(gen):
    if gen is None: return None
    else: return [gen[0]]
def use_generator(gen,obj):
    if gen is None: return id(obj)
    else:
        u = gen[0]
        gen[0] = u+1
        return u


# ==========================
# ========= DEVICE ========= 
# ==========================

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_device_and_check_all_same_device(
        model,dict_inputs,without_inp=False):
    d = None
    k = None
    print_err = lambda k1,d1,k2,d2 : raise_(
      f"Carelessness ! All inputs and parameters of the model\n"\
      f"must share the same device. Here {k1}'s device is {d1}\n"\
      f"and {k2}'s device is {d2}.")

    if not isinstance(dict_inputs,dict):
        dict_inputs = dict(enumerate(dict_inputs))

    for (key,inp) in dict_inputs.items():
        if isinstance(inp,torch.Tensor):
            if d is None: d = inp.device ; k = f"input {key}"
            else:
                if d != inp.device:
                    print_err(f"input {key}",inp.device,k,d)
    i = -1
    for p in model.parameters():
        i += 1
        if d is None: d = p.device ; k = f"{i}-th parameter"
        else:
            if d != p.device:
                print_err(f"{i}-th parameter",p.device,k,d)
    if d: return d
    elif without_inp: return get_device()
    else: raise Exception(
        "Sorry, at least one input or one parameter should be a tensor.")

# ==========================



# ==========================
# ==== Improved __eq__ =====
# ==========================

def clean__eq__(a1,a2,raise_exception=False):
    if not raise_exception: return bool(a1 == a2)
    if type(a1) != type(a2): raise Exception(
        f"{a1} and {a2} differ on type")
    if (isinstance(a1,list)
    or isinstance(a1,tuple)
    or isinstance(a1,set)):
        if len(a1) != len(a2): raise Exception(
            f"iterables diff: length diff: {len(a1)} != {len(a2)}")
        for x1,x2 in zip(a1,a2): clean__eq__(x1,x2,True)
    elif isinstance(a1,dict):
        keys1 = list(a1.keys())
        nb1 = len(keys1)
        nb2 = len(a2.keys())
        if nb1 != nb2: raise Exception(
            f"dict diff : nb of keys diff : {nb1} != {nb2}")
        for k in keys1:
            if k not in a2: raise Exception(
                f"dict diff : {k} is in dict1 but not dict2")
            clean__eq__(a1[k],a2[k],True)
    else:
        try: return a1.__eq__(a2,raise_exception=True)
        except TypeError:
            b = bool(a1 == a2)
            if not b and raise_exception: raise Exception(
                f"clean__eq__ default eq test : {a1} != {a2}")
    return True

def check_attr(o1,o2,list_attr,raise_exception=False):
    for s in list_attr:
        v1 = getattr(o1,s)
        v2 = getattr(o2,s)
        if not raise_exception:
            if v1 != v2: return False
        else: clean__eq__(v1,v2,raise_exception=True)
    return True

# ==========================



# ==========================
# == Concerning data_ptr ===
# ==========================

def has_a_data_ptr(value):
    return (
    isinstance(value,torch.Tensor)
        or
        ( ( isinstance(value,list) or isinstance(value,tuple))
            and
            isinstance(value[0],torch.Tensor))
    )


def get_data_ptr(value):
    if (isinstance(value,list)
    or  isinstance(value,tuple)):
        return get_data_ptr(value[0])
    elif isinstance(value,torch.Tensor):
        return value.data_ptr()
    else: return None

# ==========================



# ==========================
# == SAFELY USE GRAPHVIZ ===
# ==========================

def graph_render(dot,open,graph_type,render_format):
    try:
      dot.render(directory="graphviz_dir",
              format=render_format,
              quiet=True,
              view=open)
    except: print(
        f"Warning : issue with graphviz to print {graph_type}_graph, "\
        f"probably because Graphviz isn't installed on the computer "\
        f"(the software, not the python module). Normally the .gv "\
        f"has been generated, but not the .pdf",
        file = sys.stderr)

# ==========================

