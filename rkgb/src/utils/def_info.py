# ==========================
# ====== Tensor INFO =======
# ==========================

from rkgb.utils.imports import *
from rkgb.utils import small_fcts
from rkgb.utils import global_vars

# -> all the info concerning a variable/tensor which might be useful
# -> e.g. to regenerate it, using def_info.generate_val(info,device)

# attributes : 
# dtype ; ttype ; tsize
# sub_info ; requires_grad ; memsize
# is_view ; is_inplace ; 
# data_owner_name ; data_direct_parent_name

class Var_info(): # everything needed to randomly regenerate a var
    def __init__(self,
        value=None,
        is_view=False,
        is_inplace=False,
        data_owner_name=None,
        data_direct_parent_name=None
    ):
        self.is_view    = is_view
        self.is_inplace = is_inplace
        self.data_owner_name = data_owner_name
        ddpn = data_direct_parent_name
        self.data_direct_parent_name = ddpn if ddpn else data_owner_name
        # -> in case is_view or is_inplace
        if value is None:
            self.ttype = None
        else:
            if (isinstance(value,int) or
                (isinstance(value,torch.Tensor)
                and value.shape==torch.Size([]))):
                self.ttype = tt = torch.Size
            else: self.ttype = tt = type(value)
            if tt==torch.Size:
                self.tsize = (
                    value if isinstance(value,int) else value.clone())
                self.requires_grad = False
            elif isinstance(value,torch.Tensor):
                self.tsize = value.shape
                self.dtype = value.dtype
                self.requires_grad = value.requires_grad
                self.memsize = (
                    int(irotor.tensorMsize(value)))
            elif tt==tuple or tt==list:
                self.sub_info = [Var_info(y) for y in value]
            else:
                raise Exception(
                    f"The type {tt} is not supported for Var_info")

    def __eq__(self,i2,raise_exception=False):
        i1 = self
        d = small_fcts.vdir(i1)
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
        attrs = small_fcts.vdir(self)
        for attr in attrs:
            if hasattr(self,attr):
                s += f"\t{attr} = {getattr(self,attr)}\n"
        return s

    def copy(self):
        clone = Var_info()
        attrs = small_fcts.vdir(self)
        for attr in attrs:
            setattr(clone,attr,getattr(self,attr))
        clone.data_owner_name = str(self.data_owner_name)
        clone.data_direct_parent_name = str(self.data_direct_parent_name)
        return clone



def generate_val(info,device):
    tt = info.ttype
    if tt==torch.Size:
        return info.tsize
    elif tt==torch.Tensor:
        if info.dtype in global_vars.int_dtype:
            return torch.randint(128,info.tsize,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
        elif info.dtype in global_vars.bool_dtype:
            return torch.randint(2,info.tsize,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
        else: # float or complexe
            return torch.randn(info.tsize,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
    else:
        assert(tt==list or tt==tuple)
        x = [generate_val(sub_info,device) for sub_info in info.sub_info]
        return tt(x)

# ==========================


