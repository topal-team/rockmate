# ==========================
# ==== Useful functions ====
# ==========================
import inspect
from torch import Tensor

class Counter():
    def __init__(self):
        self.c = 0
    def count(self):
        self.c += 1
        return self.c
    def value(self):
        return self.c

# -> to get all the attrs except special ones
def all_non_private_attributes(c):
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
