# ==========================
# ====== Tensor INFO =======
# ==========================

import torch
from src.utils.utils import all_non_private_attributes
from src.lowlevel import constants
from src.lowlevel import measure

# -> all the info concerning a variable/tensor which might be useful
# -> e.g. to regenerate it, using def_info.generate_val(info,device)

# attributes : 
# dtype ; variable_type ; tensor_size
# sub_info ; requires_grad ; memsize
# is_view ; is_inplace ; 
# data_owner_name ; data_direct_parent_name

class VariableInfo():
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
        if data_direct_parent_name is None:
            self.data_direct_parent_name = data_owner_name
        else:
            self.data_direct_parent_name = data_direct_parent_name
        # -> in case is_view or is_inplace
        if value is None:
            self.variable_type = None
        else:
            if (isinstance(value,int) or
                (isinstance(value,torch.Tensor)
                and not value.requires_grad
                and value.shape==torch.Size([]))):
                self.variable_type = tt = torch.Size
            else: self.variable_type = tt = type(value)
            if tt==torch.Size:
                self.tensor_size = (
                    value if isinstance(value,int) else value.clone())
                self.requires_grad = False
            elif isinstance(value,torch.Tensor):
                self.variable_type = torch.Tensor
                self.tensor_size = value.shape
                self.dtype = value.dtype
                self.requires_grad = value.requires_grad
                self.memsize = (
                    int(measure.tensorMsize(value)))
            elif tt==tuple or tt==list:
                self.sub_info = [VariableInfo(y) for y in value]
                self.requires_grad = any([sub.requires_grad for sub in self.sub_info])
            else:
                raise Exception(
                    f"The type {tt} is not supported for VariableInfo")
            
    @staticmethod
    def has_a_data_ptr(value):
        return (
        isinstance(value,torch.Tensor)
            or
            ( ( isinstance(value,list) or isinstance(value,tuple))
                and
                any([VariableInfo.has_a_data_ptr(v) for v in value]))
        )

    @staticmethod
    def get_data_ptr(value):
        if isinstance(value,torch.Tensor):
            return value.data_ptr()
        elif (isinstance(value,list) or isinstance(value,tuple)):
            for v in value:
                v_ptr = VariableInfo.get_data_ptr(v)
                if not (v_ptr is None):
                    return v_ptr
            return None
        else: return None


    def __eq__(self,i2,raise_exception=False):
        i1 = self
        d = all_non_private_attributes(i1)
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
        attrs = all_non_private_attributes(self)
        for attr in attrs:
            if hasattr(self,attr):
                s += f"\t{attr} = {getattr(self,attr)}\n"
        return s

    def copy(self):
        clone = VariableInfo()
        attrs = all_non_private_attributes(self)
        for attr in attrs:
            setattr(clone,attr,getattr(self,attr))
        clone.data_owner_name = str(self.data_owner_name)
        clone.data_direct_parent_name = str(self.data_direct_parent_name)
        return clone



def generate_val(info,device):
    tt = info.variable_type
    if tt==torch.Size:
        return info.tensor_size
    elif tt==torch.Tensor:
        if info.dtype in constants.int_dtype:
            return torch.randint(128,info.tensor_size,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
        elif info.dtype in constants.bool_dtype:
            return torch.randint(2,info.tensor_size,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
        else: # float or complexe
            return torch.randn(info.tensor_size,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
    else:
        assert(tt==list or tt==tuple)
        x = [generate_val(sub_info,device) for sub_info in info.sub_info]
        return tt(x)

# ==========================


