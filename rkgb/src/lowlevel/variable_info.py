# ==========================
# ====== Tensor INFO =======
# ==========================

import torch
from src.utils.utils import all_non_private_attributes
from src.lowlevel import constants
from src.lowlevel import measure

# -> all the info concerning a variable/tensor which might be useful
# -> e.g. to regenerate it, using def_info.generate_value(info,device)

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
        is_param=False,
        data_owner_name=None,
        data_direct_parent_name=None
    ):
        self.is_view    = is_view
        self.is_inplace = is_inplace
        self.is_param   = is_param
        self.data_owner_name = data_owner_name
        if data_direct_parent_name is None:
            self.data_direct_parent_name = data_owner_name
        else:
            self.data_direct_parent_name = data_direct_parent_name
        # -> in case is_view or is_inplace
        if value is None:
            self.variable_type = None
        else:
            # 1) Find the variable_type: either a size or type(value)
            if (isinstance(value,int) or
                (isinstance(value,torch.Tensor)
                and not value.requires_grad
                and value.shape==torch.Size([]))):
                self.variable_type = torch.Size
            else: self.variable_type = type(value)
            # 2) Fill attributes: size/dtype/requires_grad etc
            if self.variable_type==torch.Size:
                self.tensor_size = (
                    value if isinstance(value,int) else value.clone())
                self.requires_grad = False
            elif isinstance(value,torch.Tensor):
                self.variable_type = torch.Tensor
                self.tensor_size = value.shape
                self.dtype = value.dtype
                self.requires_grad = value.requires_grad
                self.memsize = int(measure.tensorMsize(value))
            elif self.variable_type==tuple or self.variable_type==list:
                self.sub_info = [VariableInfo(y) for y in value]
                self.requires_grad = any([
                    sub_var.requires_grad 
                    for sub_var in self.sub_info])
            else:
                raise Exception(
                    f"The type `{self.variable_type}` "\
                    f"is not supported for VariableInfo")
            

    def generate_value(self,device):
        if self.variable_type==torch.Size:
            return self.tensor_size
        elif self.variable_type==torch.Tensor:
            if self.dtype in constants.int_dtype:
                return torch.randint(128,self.tensor_size,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    device=device)
            elif self.dtype in constants.bool_dtype:
                return torch.randint(2,self.tensor_size,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    device=device)
            else: # float or complex
                return torch.randn(self.tensor_size,
                    dtype=self.dtype,
                    requires_grad=self.requires_grad,
                    device=device)
        else:
            assert(self.variable_type==tuple or self.variable_type==list)
            value = [sub_var.generate_value(device) for sub_var in self.sub_info]
            return self.variable_type(value)


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

    @staticmethod
    def find_all_data_ptr_of_params(model : torch.nn.Module):
        all_data_ptrs = set()
        for param in model.parameters:
            data_ptr = VariableInfo.get_data_ptr(param)
            if data_ptr is not None: all_data_ptrs.add(data_ptr)


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



# ==========================


