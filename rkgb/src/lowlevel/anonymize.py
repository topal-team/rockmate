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



class AnonymousHash():
    """
    two objects are equivalent up to renaming 
    <=> they have the same AnonymousHash
    a string that contains all the information

    We can do an AnonymousHash for the following classes:
    - VariableInfo
    - SimplifiedNodeAnonymizationMaterial => SimplifiedNode
    - PartitionedCluster
    """
    @staticmethod
    def variable_info(info : VariableInfo):
        if info.variable_type is tuple or info.variable_type is list:
            return str((
                info.variable_type,
                [AnonymousHash.variable_info(sub)
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
                (ano_tar,AnonymousHash.variable_info(info)))
        for cst_ano_name,info in sn_ano_material.dict_cst_ano_name_to_variable_info.items():
            ano_repr.append(
                (cst_ano_name,AnonymousHash.variable_info(info)))
        for param_ano_name,info in sn_ano_material.dict_param_ano_name_to_variable_info.items():
            ano_repr.append(
                (param_ano_name,AnonymousHash.variable_info(info)))
        return str(ano_repr)
    
    @staticmethod
    def partitioned_cluster(p_cluster):
        sg = p_cluster.p_structure.sg
        sg_edges_labels = sg.dict_of_labels_on_edges
        dict_target_ano_id = p_cluster.p_structure.dict_target_ano_id
        dict_mt_to_sn_ano_material = p_cluster.p_structure.dict_mt_to_sn_ano_material
        hash_list = []
        # == Process each node one by one ==
        for sn_to_proceed in p_cluster.s_nodes:
            sn_ano_material : SimplifiedNodeAnonymizationMaterial \
                = dict_mt_to_sn_ano_material[sn_to_proceed.mt]
            # 1) Write dependency edges: who sn depends on and what it uses from it
            hash_edges = []
            for req_sn in sn_to_proceed.deps:
                if req_sn not in p_cluster.s_nodes: continue
                hash_used_targets = []
                for used_target in sg_edges_labels[(req_sn,sn_to_proceed)]:
                    hash_used_targets.append((
                        sn_ano_material.dict_tar_to_ano_nb[used_target], # => its ano_nb in sn_to_proceed's code
                        req_sn.all_targets.index(used_target))) # => its index in req_sn
                hash_used_targets.sort(key = lambda c : c[0]) # ie target's ano_nb in sn_to_proceed's code
                hash_edges.append((
                    p_cluster.s_nodes.index(req_sn), # who sn depends on 
                    hash_used_targets # what targets it uses from it
                ))
            hash_edges.sort(key = lambda c : c[0]) # ie s_nodes.index(req_sn)

            # 2) In case sn_to_proceed is a first node of the cluster
            # => hash its interactions with input nodes
            hash_inputs = []
            if sn_to_proceed in p_cluster.first_snodes:
                for req_input_sn in p_cluster.dict_first_sn_to_required_inputs_sn[sn_to_proceed]:
                    hash_used_targets = []
                    for used_target in sg_edges_labels[(req_input_sn,sn_to_proceed)]:
                        hash_used_targets.append((
                            sn_ano_material.dict_tar_to_ano_nb[used_target], # => its ano_nb in sn_to_proceed's code
                            dict_target_ano_id[used_target] # => target's equivalence class: based on its VariableInfo
                        ))
                    hash_used_targets.sort(key = lambda c : c[0]) # ie target's ano_nb in sn_to_proceed's code
                    hash_inputs.append((
                        p_cluster.input_snodes.index(req_input_sn),
                        hash_used_targets
                    ))
                hash_inputs.sort(key = lambda c : c[0])
                # Note: we don't rely on req_input_sn's equivalence class
                # instead we look at the targets we use and their types
                # ie VariableInfo equivalence class;

            # 3) In case its an output node of the cluster
            # => hash which targets are returned
            hash_outputs = []
            if sn_to_proceed in p_cluster.output_snodes:
                for sent_target in p_cluster.dict_output_mt_to_targets_sent[sn_to_proceed.mt]:
                    hash_outputs.append(sn_to_proceed.all_targets.index(sent_target))
                hash_outputs.sort()
                # => Simply return all outputs' index

            # 4) The final hash of sn_to_proceed:
            hash_list.append((
                sn_ano_material.anonymous_id,
                hash_edges,
                hash_inputs,
                hash_outputs
            ))

        return str(hash_list)


    
    @staticmethod
    def hash(object):
        if isinstance(object,VariableInfo):
            return AnonymousHash.variable_info(object)
        elif isinstance(object,SimplifiedNodeAnonymizationMaterial):
            return AnonymousHash.simplified_node_anonymization_material(object)
        elif type(object).__name__ == "PartitionedCluster":
            return AnonymousHash.partitioned_cluster(object)
        else:
            raise Exception(f"AnonymousHash unsupported type: {type(object)}")



def build_anonymous_equivalence_classes(
        sg : SimplifiedGraph,
        original_mod : torch.nn.Module):
    """
    Return:
     - dict_target_ano_id:
        Two targets have the same ano_id if their 
        VariableInfo are equal up to anonymization.
    
     - dict_mt_to_sn_ano_material:
        Two nodes have the same ano_id if their
        AnonymizedHash are equal: ie
        1) Their codes are equal up anonymization
        2) All their variables/constants/params have
        equal VariableInfo, ie equal type, shape, etc
    """
    # 1) Build node equivalence classes:
    # => Assign to each node its anonymous material and an anonymous id
    dict_mt_to_sn_ano_material = dict()
    dict_sn_ano_hash_to_sn_ano_id = dict()
    counter_nb_unique_snodes = Counter()
    for sn_to_proceed in [sg.init_node] + sg.nodes:
        sn_ano_material = SimplifiedNodeAnonymizationMaterial(sn_to_proceed,sg,original_mod)
        sn_ano_hash = AnonymousHash.hash(sn_ano_material)
        if sn_ano_hash in dict_sn_ano_hash_to_sn_ano_id:
            sn_ano_material.anonymous_id \
                = dict_sn_ano_hash_to_sn_ano_id[sn_ano_hash]
        else:
            sn_ano_material.anonymous_id \
                = dict_sn_ano_hash_to_sn_ano_id[sn_ano_hash] \
                = counter_nb_unique_snodes.count()
        dict_mt_to_sn_ano_material[sn_to_proceed.main_target] = sn_ano_material

    # 2) Build target equivalence classes
    # => Assign to each target an anonymous id
    dict_target_ano_id = dict()
    dict_info_ano_hash_to_tar_ano_id = dict()
    counter_nb_unique_targets = Counter()
    for target,info in sg.dict_info.items():
        target_ano_hash = AnonymousHash.hash(info)
        if target_ano_hash in dict_info_ano_hash_to_tar_ano_id:
            dict_target_ano_id[target] \
                = dict_info_ano_hash_to_tar_ano_id[target_ano_hash]
        else:
            dict_target_ano_id[target] \
                = dict_info_ano_hash_to_tar_ano_id[target_ano_hash] \
                = counter_nb_unique_targets.count()
    
    return (
        dict_target_ano_id,
        dict_mt_to_sn_ano_material
    )



def sort_inputs_mt(p_cluster,input_snodes):
    sg = p_cluster.p_structure.sg
    dict_mt_to_sn_ano_material = p_cluster.p_structure.dict_mt_to_sn_ano_material
    # To use at the very end of cluster.compute_interfaces()
    # We sort inputs_mt based on the targets the sent
    # There is no optimal order => we just want to be sure
    # to equivalent clusters to have equivalent orders
    dict_input_sn_to_repr = dict()
    for input_sn in input_snodes:
        repr = []
        users_with_index = [
            (user_sn,p_cluster.first_snodes.index(user_sn))
            for user_sn in p_cluster.dict_input_sn_to_users_sn[input_sn]
        ]
        users_with_index = sorted(users_with_index,key = lambda c : c[1])
        for user_sn,user_index in users_with_index:
            user_ano_material : SimplifiedNodeAnonymizationMaterial \
                = dict_mt_to_sn_ano_material[user_sn.mt]
            targets_used = sg.dict_of_labels_on_edges[(input_sn,user_sn)]
            targets_numbers = sorted(
                user_ano_material.dict_tar_to_ano_nb[tar]
                for tar in targets_used)
            repr.append((user_index,targets_numbers))
        dict_input_sn_to_repr[input_sn] = str(repr)
    return sorted(input_snodes, 
        key = lambda input_sn : dict_input_sn_to_repr[input_sn])




class ClusterTranslator():
    """
    Translate SimplifiedNode/target/ComputationNode/AllocationNode/Node's name
    to an anonymized object, and then reverse translate.
    The idea is to translate with the Translator of one cluster 
    and then reverse translate for an equivalent cluster.
    """
    def __init__(self,partitioned_cluster):
        self.dict_to_ano = dict()
        self.dict_from_ano = dict()
        # Take care of anonymizing targets and SimplifiedNode
        # Computation/Allocation nodes will be done latter

        # As we assume equivalent graphs
        # In particular they have equivalent topological order
        for index,sn in enumerate(partitioned_cluster.s_nodes):
            self.dict_to_ano[sn.main_target] = index
            self.dict_from_ano[index] = sn.main_target
            ano = ("sn",index)
            self.dict_to_ano[sn] = ano
            self.dict_from_ano[ano] = sn

        # We assume self.inputs_mt is a properly sorted list
        # such that equivalent graphs have equal inputs_mt order
        # up to anonymization
        for index,input_mt in enumerate(partitioned_cluster.inputs_mt):
            ano = ("input",index)
            self.dict_to_ano[input_mt] = ano
            self.dict_from_ano[ano] = input_mt

    def enrich_with_cnodes_and_anodes(self,hierarchical_cluster):
        # 1) Computation Nodes
        for cnode in hierarchical_cluster.list_cnodes:
            if cnode is hierarchical_cluster.loss_cnode:
                ano = ("cnode","loss")
            else:
                mt_ano = self.dict_to_ano[cnode.mt]
                ano = ("cnode",cnode.is_fwd,mt_ano)
            self.dict_to_ano[cnode] = ano
            self.dict_from_ano[ano] = cnode

        # 2) Allocation Nodes
        for anode in hierarchical_cluster.list_anodes:
            mt_ano = self.dict_to_ano[anode.mt]
            ano = ("anode",anode.allocation_type,mt_ano)
            self.dict_to_ano[cnode] = ano
            self.dict_from_ano[ano] = cnode

        # 3) Node names
        for node_name,node in hierarchical_cluster.dict_nodes.items():
            node_ano = self.dict_to_ano[node]
            ano = ("name",)+node_ano
            self.dict_to_ano[node_name] = ano
            self.dict_from_ano[ano] = node_name
    
            
    def to_ano(self,object):
        return self.dict_to_ano[object]
    def from_ano(self,ano):
        return self.dict_from_ano[ano]