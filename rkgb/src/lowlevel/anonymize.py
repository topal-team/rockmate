# Help recognize similar nodes, => similar clusters
# so we don't solve twice equivalent instances

import ast
import torch
from src.utils.utils import Counter
from src.core import base
from src.lowlevel.variable_info import VariableInfo
from src.core.simplified import SimplifiedGraph, SimplifiedNode
from src.core.backward import ComputationNode, AllocationNode


class SimplifiedNodeAnonymizationMaterial():
    anonymous_id : int = None # will be set at higher level
    anonymized_code : str = None
    dict_tar_to_ano_nb : dict[str, int] = None
    dict_tar_to_ano_tar : dict[str, str] = None
    dict_cst_name_to_ano_name : dict[str, str] = None
    dict_param_name_to_ano_name : dict[str, str] = None
    dict_ano_tar_to_variable_info : dict[str, VariableInfo] = None
    dict_cst_ano_name_to_variable_info : dict[str, VariableInfo] = None
    dict_param_ano_name_to_variable_info : dict[str, VariableInfo] = None

    def __init__(self,
            sn_to_proceed : SimplifiedNode, 
            simplified_graph : SimplifiedGraph, 
            original_mod : torch.nn.Module):
        # FIRST : read the code and collect all
        # the variables/constants/params mentioned
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

        # =======
        # SECOND : associate a number to 
        # each variables/constants/params  
        # => anonymized name ~ f"__{ano_nb}_ano"
        self.dict_tar_to_ano_nb = dict()
        self.dict_tar_to_ano_tar = dict()
        self.dict_cst_name_to_ano_name = dict()
        self.dict_param_name_to_ano_name = dict()
        self.dict_ano_tar_to_variable_info = dict()
        self.dict_cst_ano_name_to_variable_info = dict()
        self.dict_param_ano_name_to_variable_info = dict()
        # Associate numbers to *variables*
        all_real_vars = sorted(all_real_vars,key = base.Node.get_num_tar)
        nb_var = 0
        for real_name in all_real_vars:
            nb_var += 1
            ano_tar = f"__{nb_var}_ano"
            self.dict_tar_to_ano_tar[real_name] = ano_tar
            self.dict_tar_to_ano_nb [real_name] = nb_var
            self.dict_ano_tar_to_variable_info[ano_tar] \
                = simplified_graph.dict_info[real_name]

        # Associate numbers to *constants*
        all_real_cst = sorted(all_real_cst,key = base.Node.get_num_tar)
        nb_cst = 0
        for cst_real_name in all_real_cst:
            cst_value = simplified_graph.dict_constants[cst_real_name]
            nb_cst += 1
            ano_name = f"_cst_{nb_cst}_ano"
            self.dict_cst_name_to_ano_name[cst_real_name] = ano_name
            self.dict_cst_ano_name_to_variable_info[ano_name] = VariableInfo(cst_value)

        # Associate numbers to *constants*
        nb_param = 0
        for param_full_name in all_real_params: # strings
            # -> e.g. param_full_name = "self.layer1.weight"
            param_value = eval(param_full_name,{"self":original_mod},{})
            nb_param += 1
            ano_name = f"self.param_{nb_param}"
            self.dict_param_name_to_ano_name[param_full_name] = ano_name
            self.dict_param_ano_name_to_variable_info[ano_name] = VariableInfo(param_value)
                
        # =======
        # THIRD: replace all names by their anonymized version
        code_to_proceed = sn_to_proceed.get_code()
        for tar,ano_tar in self.dict_tar_to_ano_tar.items():
            code_to_proceed = code_to_proceed.replace(tar,ano_tar)
        for cst_name,ano_name in self.dict_cst_name_to_ano_name.items():
            code_to_proceed = code_to_proceed.replace(cst_name,ano_name)
        for param_name,ano_name in self.dict_param_name_to_ano_name.items():
            code_to_proceed = code_to_proceed.replace(param_name,ano_name)
        self.anonymized_code = code_to_proceed
    # ============================



class AnonymousReprString():
    """
    two objects are equivalent up to renaming 
    <=> they have the same AnonymousReprString
    a string that contains all the information

    We can do an AnonymousReprString for the following classes:
    - VariableInfo
    - SimplifiedNodeAnonymizationMaterial => SimplifiedNode
    - PartitionedCluster
    """
    @staticmethod
    def variable_info(info : VariableInfo):
        if info.variable_type is tuple or info.variable_type is list:
            return str((
                info.variable_type,
                [AnonymousReprString.variable_info(sub)
                 for sub in info.sub_info]
            ))
        else:
            return str((
                info.variable_type,
                info.dtype if hasattr(info,"dtype") else None,
                info.tensor_size if hasattr(info,"tensor_size") else None,
                info.requires_grad if hasattr(info,"requires_grad") else None,
                info.memsize if hasattr(info,"memsize") else None,
            ))
        
    @staticmethod
    def simplified_node_anonymization_material(
            sn_ano_material : SimplifiedNodeAnonymizationMaterial):
        ano_repr = [sn_ano_material.anonymized_code]
        for ano_tar,info in sn_ano_material.dict_ano_tar_to_variable_info.items():
            ano_repr.append(
                (ano_tar,AnonymousReprString.variable_info(info)))
        for cst_ano_name,info in sn_ano_material.dict_cst_ano_name_to_variable_info.items():
            ano_repr.append(
                (cst_ano_name,AnonymousReprString.variable_info(info)))
        for param_ano_name,info in sn_ano_material.dict_param_ano_name_to_variable_info.items():
            ano_repr.append(
                (param_ano_name,AnonymousReprString.variable_info(info)))
        return str(ano_repr)
    
    @staticmethod
    def get(object):
        if isinstance(object,VariableInfo):
            return AnonymousReprString.variable_info(object)
        elif isinstance(object,SimplifiedNodeAnonymizationMaterial):
            return AnonymousReprString.simplified_node_anonymization_material(object)
        else:
            raise Exception(f"AnonymousReprString unsupported type: {type(object)}")



def build_anonymous_equivalence_classes(
        sg : SimplifiedGraph,
        original_mod : torch.nn.Module):
    """
    Return:
     - dict_target_anonymous_id:
        Two targets have the same ano_id if their 
        VariableInfo are equal up to anonymization.
    
     - dict_main_target_to_node_anonymous_material:
        Two nodes have the same ano_id if their
        AnonymizedReprString are equal: ie
        1) Their codes are equal up anonymization
        2) All their variables/constants/params have
        equal VariableInfo, ie equal type, shape, etc
    """
    # 1) Build node equivalence classes:
    # => Assign to each node its anonymous material and an anonymous id
    dict_main_target_to_node_anonymous_material = dict()
    dict_sn_ano_repr_string_to_anonymous_id = dict()
    counter_nb_unique_snodes = Counter()
    for sn_to_proceed in [sg.init_node] + sg.nodes:
        sn_ano_material = SimplifiedNodeAnonymizationMaterial(sn_to_proceed,sg,original_mod)
        sn_ano_repr_string = AnonymousReprString.get(sn_ano_material)
        if sn_ano_repr_string in dict_sn_ano_repr_string_to_anonymous_id:
            sn_ano_material.anonymous_id \
                = dict_sn_ano_repr_string_to_anonymous_id[sn_ano_repr_string]
        else:
            sn_ano_material.anonymous_id \
                = dict_sn_ano_repr_string_to_anonymous_id[sn_ano_repr_string] \
                = counter_nb_unique_snodes.count()
        dict_main_target_to_node_anonymous_material[sn_to_proceed.main_target] = sn_ano_material

    # 2) Build target equivalence classes
    # => Assign to each target an anonymous id
    dict_target_anonymous_id = dict()
    dict_info_ano_repr_string_to_anonymous_id = dict()
    counter_nb_unique_targets = Counter()
    for target,info in sg.dict_info.items():
        target_ano_repr_string = AnonymousReprString.get(info)
        if target_ano_repr_string in dict_info_ano_repr_string_to_anonymous_id:
            dict_target_anonymous_id[target] \
                = dict_info_ano_repr_string_to_anonymous_id[target_ano_repr_string]
        else:
            dict_target_anonymous_id[target] \
                = dict_info_ano_repr_string_to_anonymous_id[target_ano_repr_string] \
                = counter_nb_unique_targets.count()
    
    return (
        dict_target_anonymous_id,
        dict_main_target_to_node_anonymous_material
    )



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