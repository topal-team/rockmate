from .utils import *

# ==========================
# ====== Tensor INFO =======
# ==========================

# -> all the info concerning a variable/tensor which might be useful
# -> e.g. to regenerate it, using def_info.generate_var(info,device)

# attributes : 
# dtype ; ttype ; tsize
# sub_info ; requires_grad ; memsize
# is_view ; is_inplace ; 
# data_owner_name ; data_direct_parent_name

class Var_info(): # everything needed to randomly regenerate a var
    def __init__(self,
        value,
        is_view=False,
        is_inplace=False,
        data_owner_name=None,
        data_direct_parent_name=None):
        self.is_view    = is_view
        self.is_inplace = is_inplace
        self.data_owner_name = data_owner_name
        ddpn = data_direct_parent_name
        self.data_direct_parent_name = ddpn if ddpn else data_owner_name
        # -> in case is_view or is_inplace
        if (isinstance(value,int) or
            (isinstance(value,torch.Tensor)
            and value.shape==torch.Size([]))):
            self.ttype = tt = torch.Size
        else: self.ttype = tt = type(value)
        if tt==torch.Size:
            self.tsize = int(value)
            self.requires_grad = False
        elif isinstance(value,torch.Tensor):
            self.tsize = value.shape
            self.dtype = value.dtype
            self.requires_grad = value.requires_grad
            self.memsize = MemSize(int(tensorMsize(value)))
        elif tt==tuple or tt==list:
            self.sub_info = [Var_info(y) for y in value]
        else:
            raise Exception(f"The type {tt} is not supported for Var_info")

    def __eq__(self,i2,raise_exception=False):
        i1 = self
        d = vdir(i1)
        for s in d:
            v1 = getattr(i1,s)
            v2 = getattr(i2,s)
            if v1 != v2:
                if raise_exception: raise Exception(
                    f"Info diff on attr : {s} : {v1} != {v2}")
                return False
        return True

    def __str__(self):
        s = ""
        attrs = vdir(self)
        for attr in attrs:
            if hasattr(self,attr):
                s += f"\t{attr} = {getattr(self,attr)}\n"
        return s


def generate_val(info,device):
    tt = info.ttype
    if tt==torch.Size:
        return info.tsize
    elif tt==torch.Tensor:
        return torch.ones(info.tsize,
            dtype=info.dtype,
            requires_grad=info.requires_grad,
            device=device)
    else:
        assert(tt==list or tt==tuple)
        x = [generate_val(sub_info,device) for sub_info in info.sub_info]
        return tt(x)

# ==========================


