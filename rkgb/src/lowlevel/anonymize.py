# Help recognize similar nodes, => similar clusters
# so we don't solve twice equivalent instances

import ast
import torch
from src.core import base
from src.lowlevel.variable_info import VariableInfo
from src.core.simplified import SimplifiedGraph, SimplifiedNode
from src.core.backward import ComputationNode, AllocationNode


class SimplifiedNodeAnonymizedInfo():
    ano_id : int = None # will be set at higher level
    ano_str_code : str = None
    dict_tar_to_ano_nb : dict[str, int] = None
    dict_tar_to_ano_tar : dict[str, str] = None
    dict_cst_name_to_ano_name : dict[str, str] = None
    dict_param_name_to_ano_name : dict[str, str] = None
    dict_ano_tar_to_variable_info : dict[str, VariableInfo] = None
    dict_ano_cst_to_variable_info : dict[str, VariableInfo] = None
    dict_ano_param_to_variable_info : dict[str, VariableInfo] = None

    # =====================================================================
    def __init__(self,
            sn_to_proceed : SimplifiedNode, 
            simplified_graph : SimplifiedGraph, 
            original_mod : torch.nn.Module):
        # = FIRST : read the code and collect all    =
        # = the variables/constants/params mentioned =
        all_real_vars   = []
        all_real_cst    = []
        all_real_params = []
        def handle_str(real_str):
            if (real_str[:2] == "__"
            and not real_str in all_real_vars):
                all_real_vars.append(real_str)
            elif (real_str[:5] == "self."
            or real_str[:5] == "self["
            or real_str[:13] == "getattr(self."
            and not real_str in all_real_params):
                all_real_params.append(real_str)
            elif (real_str[:5] == "_cst_"
            and not real_str in all_real_cst):
                all_real_cst.append(real_str)
        def search_through(a):
            if isinstance(a,ast.AST):
                if isinstance(a,ast.Name):
                    handle_str(a.id)
                else:
                    for s in a._fields:
                        try: search_through(getattr(a,s))
                        except: pass
            elif isinstance(a,str): handle_str(a)
            elif hasattr(a,"__iter__"):
                for sub_a in a: search_through(sub_a)

        search_through(sn_to_proceed.get_code_ast())

        # = SECOND : associate a number to 
        # = each variables/constants/params  
        # => anonymized name ~ f"__{ano_nb}_ano"
        self.dict_tar_to_ano_nb = dict()
        self.dict_tar_to_ano_tar = dict()
        self.dict_cst_name_to_ano_name = dict()
        self.dict_param_name_to_ano_name = dict()
        self.dict_ano_tar_to_variable_info = dict()
        self.dict_ano_cst_to_variable_info = dict()
        self.dict_ano_param_to_variable_info = dict()
        # Associate numbers to *variables*
        all_real_vars = sorted(all_real_vars,key = base.Node.get_num_tar)
        nb_var = 0
        for real_name in all_real_vars:
            nb_var += 1
            anonymized_name = f"__{nb_var}_ano"
            self.dict_tar_to_ano_tar[real_name] = anonymized_name
            self.dict_tar_to_ano_nb [real_name] = nb_var
            self.dict_ano_tar_to_variable_info[anonymized_name] \
                = simplified_graph.dict_info[real_name]
            # -> We will keep only basic attributes of VariableInfo

        # Associate numbers to *constants*
        all_real_cst = sorted(all_real_cst,key = base.Node.get_num_cst)
        nb_cst = 0
        for cst_real_name in all_real_cst:
            value = simplified_graph.dict_constants[cst_real_name]
            nb_cst += 1
            acst = f"_cst_{nb_cst}_ano"
            self.dict_cst_name_to_ano_name[cst_real_name] = acst
            self.dict_ano_cst_to_variable_info[acst] = VariableInfo(value)

        # Associate numbers to *constants*
        nb_param = 0
        for param_full_name in all_real_params: # strings
            # -> e.g. param_full_name = "self.layer1.weight"
            param_value = eval(param_full_name,{"self":original_mod},{})
            nb_param += 1
            aparam = f"self.param_{nb_param}"
            self.dict_param_name_to_ano_name[param_full_name] = aparam
            self.dict_ano_param_to_variable_info[aparam] = VariableInfo(param_value)
                
        # =============================
        # === THIRD: build ano code ===
        code_to_proceed = sn.get_code()
        for tar,atar in self.dict_tar_to_ano_tar.items():
            str_code = str_code.replace(tar,atar)
        for cst,acst in self.dict_cst_name_to_ano_name.items():
            str_code = str_code.replace(cst,acst)
        for param,aparam in dict_param_aparam.items():
            str_code = str_code.replace(param,aparam)
        self.ano_code = str_code
    # ============================


    # ============================
    @staticmethod
    def make_charac_info(info : VariableInfo):
        if info.variable_type is tuple or info.variable_type is list:
            return (
                info.variable_type,
                [SimplifiedNodeAnonymizedInfo.make_charac_info(sub) for sub in info.sub_info]
            )
        else:
            return (
                info.variable_type,
                info.dtype if hasattr(info,"dtype") else None,
                info.tensor_size if hasattr(info,"tensor_size") else None,
                info.requires_grad if hasattr(info,"requires_grad") else None,
                info.memsize if hasattr(info,"memsize") else None,
            )
    # ============================

    # ============================
    def make_charac_string(self):
        charac_list = [self.ano_code]
        for atar,info in self.dict_ano_tar_to_variable_info.items():
            charac_list.append((atar,SimplifiedNodeAnonymizedInfo.make_charac_info(info)))
        for acst,info in self.dict_ano_cst_to_variable_info.items():
            charac_list.append((acst,SimplifiedNodeAnonymizedInfo.make_charac_info(info)))
        for aparam,info in self.dict_ano_param_to_variable_info.items():
            charac_list.append((aparam,SimplifiedNodeAnonymizedInfo.make_charac_info(info)))
        return str(charac_list)
    # ============================


def build_anonymized_nodes_equivalence_classes(
        simplified_graph : SimplifiedGraph):
    """
    Return:
        dict_target_to_anonymized_target_id
        dict_main_target_to_anonymized_node_id
    """
    dict_target_to_anonymized_target_id = dict()
    dict_main_target_to_anonymized_node_id = dict()



class ClusterTranslator():
    dict_mt_to_ano_pair : dict[str, tuple[int,int]] = None
    dict_sn_to_ano_pair : dict[SimplifiedNode, tuple[int,int]] = None
    dict_ano_pair_to_sn : dict[tuple[int,int], SimplifiedNode] = None
    dict_kcn_to_ano_triplet : dict[ComputationNode, tuple[str,int,int]] = None
    dict_kdn_to_ano_triplet : dict[AllocationNode, tuple[str,int,int]] = None
    dict_ano_triplet_to_kcn : dict[tuple[str,int,int], ComputationNode] = None
    dict_ano_triplet_to_kdn : dict[tuple[str,int,int], AllocationNode] = None
    dict_name_to_ano_triplet : dict = None
    dict_ano_triplet_to_name : dict = None
    def __init__(self):
        pass