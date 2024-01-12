# =====================
# = INSPECTION OF KCN =
# =====================

import sys
import numpy as np
import gc
import torch
from src.lowlevel import ast_add_on
from src.lowlevel import constants
from src.lowlevel import measure
from src.lowlevel.variable_info import VariableInfo
from src.core import base
from src.core.simplified import SimplifiedNode,SimplifiedGraph




# ======================






# ==========================
# ======= INSPECTION =======
# ==========================

class FakeMod():
    def __init__(self):
        self.__list__ = []

    def __getitem__(self, i):
        if i>=len(self.__list__):
            self.__list__ += [FakeMod() for _ in range((i-len(self.__list__)+1))]
        return self.__list__[i]
    def __setitem__(self, i, value):
        if i>=len(self.__list__):
            self.__list__ += [FakeMod() for _ in range((i-len(self.__list__)+1))]
        self.__list__[i] = value

    def __setattr__(self, name: str, value):
        self.__dict__[name] = value
    def __getattr__(self, name):
        if name not in self.__dict__:
            self.__setattr__(name, FakeMod())
        return self.__dict__[name]


class InspectionResult():
    def __init__(self):
        self.relevant= False # -> turn True if result of inspection
        self.mem_overhead_fwd = 0
        self.mem_overhead_bwd = 0
        self.mem_run_fwd  = 0
        self.mem_run_bwd  = 0
        self.mem_fgt_fwd  = 0
        self.mem_fgt_bwd  = 0
        self.time_fwd = 0
        self.time_bwd = 0


class Inspector():
    """
    Use Inspector.generate_global_env() 
    and Inspector.generate_local_env()
    to get fresh environments where to run inspection.
    """
    @staticmethod
    def generate_global_env(
            simplified_graph : SimplifiedGraph,
            inspection_device : torch.device):
        our_global = globals().copy()
        our_global["device"] = inspection_device
        for cst_name,cst_value in simplified_graph.dict_constants.items():
            our_global[cst_name] = cst_value.to(inspection_device)
        # our_global["self"] = original_mod
        # this time we won't put the whole model in the env
        # instead we create local FakeMod
        return our_global
    
    @staticmethod
    def aux_generate_a_parameter_locally(
        param_node : base.ParameterNode,
        our_global, tmp_local,
        original_mod,
        inspection_device
    ):
        param_value = torch.nn.Parameter(
            param_node.get_value(original_mod).to(inspection_device))
        tmp_local["all_parameters"].add(param_value)
        tmp_local["__value"] = param_value
        exec(f"{param_node.param_str} = __value ; {param_node.get_code()}",
            our_global, tmp_local)

    @staticmethod
    def generate_local_env(
            sn_to_proceed : SimplifiedNode,
            simplified_graph : SimplifiedGraph,
            our_global : dict,
            original_mod : torch.nn.Module,
            inspection_device : torch.device):
        tmp_local = dict()
        tmp_local["self"] = FakeMod()
        tmp_local["all_parameters"] = set() # to find them easily
        all_inputs = (
            simplified_graph.original_mod_input_targets
            + simplified_graph.input_targets)
        tmp_local["all_input_values"] = set()
        # 1) Do we need to run the init_code:
        # - Generating the sizes related to init_code is free
        # so we can do it anyway, but if we require a tensor
        # for the moment I generate all the real inputs and then
        # run the init_code, as the tensor we need may be a view
        # TO IMPROVE ? Generate exactly the tensors needed
        init_node = simplified_graph.init_node
        if (
                ((sn_to_proceed,init_node)
                in simplified_graph.dict_of_labels_on_edges)
        and 
            any(
                simplified_graph.dict_info[needed_input].variable_type 
                is torch.Tensor
                for needed_input 
                in simplified_graph.dict_of_labels_on_edges[
                    (sn_to_proceed,init_node)
        ])):
            for inp in simplified_graph.original_mod_input_targets:
                inp_info = simplified_graph.dict_info[inp]
                tmp_local[inp] = inp_info.generate_value(inspection_device)
            for param_node in init_node.required_parameter_nodes:
                Inspector.aux_generate_a_parameter_locally(
                    param_node,our_global,tmp_local,
                    original_mod,inspection_device)
            exec(
                init_node.get_code(force_special_kwargs=True),
                our_global,tmp_local)
            for inp in all_inputs:
                tmp_local["all_input_values"].add(tmp_local[inp])
        else:
            # We don't need any tensor:
            # we generate sizes anyway as they come free 
            for inp in all_inputs:
                inp_info = simplified_graph.dict_info[inp]
                if inp_info.variable_type is not torch.Tensor:
                    inp_value = inp_info.generate_value(inspection_device)
                    tmp_local[inp] = inp_value
                    tmp_local["all_input_values"].add(inp_value)

        # 2) Generate required parameters
        for param_node in sn_to_proceed.required_parameter_nodes:
            Inspector.aux_generate_a_parameter_locally(
                param_node,our_global,tmp_local,
                original_mod,inspection_device)

        # 3) Generate all the deps
        list_nodes_to_generate = list(sn_to_proceed.deps)
        set_nodes_to_generate = set(list_nodes_to_generate)
        while list_nodes_to_generate != []:
            # Get next node to generate
            sn_to_generate : SimplifiedNode = list_nodes_to_generate.pop(0)
            if sn_to_generate is init_node: # TO REMOVE
                raise Exception("init_node in sn.deps ???")
            # Check if it's `sn`'s turn:
            # if some of the deps of sn are in the waiting list
            # ie we plan to properly generate them (because they
            # are also in main_sn's deps) it's better to wait.
            # But note that we don't add any additional node to 
            # the waiting list. So latter on, for sn'deps which
            # aren't main_sn'deps: we will just generate them on the fly.
            if set(sn_to_generate.deps).intersection(set_nodes_to_generate) != set():
                list_nodes_to_generate.append(sn_to_generate) # not his turn yet
                continue
            else:
                set_nodes_to_generate.remove(sn_to_generate)

            # We are ready to generate sn:
            # - First we create the main_target value based on info
            # - Then we run the body_code to generate views / sizes
            main_value = sn_to_generate.info.generate_value(inspection_device)
            # Some operations are impossible over leaf tensors 
            # in term of grad_fn. So we have to clone them :
            if isinstance(main_value,torch.Tensor):
                main_value = main_value.clone()
            tmp_local[sn_to_generate.main_target] = main_value
            
            # To run the body code we may need some dependencies to be
            # in tmp_local (e.g. sizes): so we create them on the fly
            # Note: a dependency of sn_to_generate which also happens to
            # be a dependency of sn_to_proceed, isn't created from info
            # but had already been generated in this while loop.
            body_code = ast_add_on.make_str_list_assign(
                sn_to_generate.body_code, force_special_kwargs=True)
            for body_target in sn_to_generate.all_targets:
                if body_target is sn_to_generate.main_target: continue
                for req_param_node in simplified_graph.dict_target_to_direct_parameter_deps[body_target]:
                    Inspector.aux_generate_a_parameter_locally(
                        req_param_node,our_global,tmp_local,
                        original_mod,inspection_device)
                for req_var_target in simplified_graph.dict_target_to_direct_variable_deps[body_target]:
                    req_var_info = simplified_graph.dict_info[req_var_target]
                    tmp_local[req_var_target] = req_var_info.generate_value(inspection_device)
            exec(body_code,our_global,tmp_local)
        return tmp_local

# ======================
    
class InspectorDefault(Inspector):
    def __init__(self,
            sn_to_proceed : SimplifiedNode,
            inspection_device,
            our_global, 
            tmp_local,
            timer : measure.Timer, 
            memory_tracker : measure.MemoryTracker,
            # TO CHANGE:
            simplified_graph,
            original_mod):
        self.sn_to_proceed = sn_to_proceed
        self.inspection_device = inspection_device
        self.our_global = our_global
        self.tmp_local = tmp_local
        self.timer = timer
        self.memory_tracker = memory_tracker
        # For debugging it's helpful to store all results and what we execute
        self.inspection_result = InspectionResult()
        self.code_run_fwd = sn_to_proceed.get_code(force_special_kwargs=True)
        target = sn_to_proceed.main_target
        self.code_run_bwd = f"{target}.backward({target}.grad)"
        self.code_del_all_targets = "; ".join(
            [f"del {target}" for target in self.sn_to_proceed.tensor_targets])
        # TO CHANGE:
        self.simplified_graph = simplified_graph
        self.original_mod = original_mod

    def inspect(self):

        self.inspection_result.relevant = True
        return self.inspection_result
        
    # ==================================
    # == ALL STEPS WE WANT TO MEASURE ==
    # FORWARD:
    def func_run_fwd(self):
        exec(self.code_run_fwd, self.our_global, self.tmp_local)

    def func_fgt_fwd(self):
        for target in self.sn_to_proceed.tensor_targets:
            value = self.tmp_local[target]
            value.data = torch.zeros(0,device=self.inspection_device)
            if value._base is not None:
                value._base.data = torch.empty(0,device=self.inspection_device)

    def func_del_fwd(self):
        exec(self.code_del_all_targets, self.our_global, self.tmp_local)

    # BACKWARD:
    def func_prepare_bwd(self):
        # TO CHANGE: I think it's not efficient to regenerate
        # a new tmp_local before every backward run. 
        # Instead it should be enough to set the gradients 
        # of all dependencies and parameters to None.
        self.tmp_local = self.generate_local_env(
            self.sn_to_proceed,
            self.simplified_graph,
            self.our_global,
            self.original_mod,
            self.inspection_device
        )
        # TO REPLACE by: self.func_fgt_bwd()
        self.func_run_fwd()
        self.tmp_local[self.sn_to_proceed.main_target].grad = (
            self.sn_to_proceed.info.generate_value(self.inspection_device)
        )

    def func_run_bwd(self):
        exec(self.code_run_bwd, self.our_global, self.tmp_local)

    def func_fgt_bwd(self):
        for req_sn in self.sn_to_proceed.deps:
            for req_target in req_sn.tensor_targets:
                self.tmp_local[req_target].grad = None
        all_required_params = self.tmp_local["all_parameters"]
        for param_value in all_required_params:
            param_value.grad = None
    # ==================================
            
        



# ======================
# TRACE GRAD_FN TO SEE WHAT IS NEEDED TO BWD
# -> Real dependencies
# -> Phantoms
# ======================

def trace_grad_fn(
        grad_fn,
        main_target="var",
        all_parameter_values=set(),
        all_input_values=set()):
    """
    Open grad_fn, looking after all the tensors linked in it
    => Parameters / 'saved_tensors' / explicit variables.
    Auxiliary function for "get_relevant_dependencies_via_grad_fn".
    But it can also be used to play with grad_fn and do some tests.

    It checks all attributes of grad_fn, and then open 
    grad_fn.next_functions to trace the whole backward tree.
    """
    explicit_vars  = set() # set of Tensors
    saved_tensors = set() # set of (name * value)
    def trace(current_grad_fn,path_from_the_origin):
        if hasattr(current_grad_fn,"variable"):
            explicit_vars.add(current_grad_fn.variable)
        for attr in dir(current_grad_fn):
            attr_value = getattr(current_grad_fn,attr)
            if (attr != "variable" 
            and isinstance(attr_value,torch.Tensor)
            and not attr_value in all_parameter_values
            and not attr_value in all_input_values):
                path_str = [
                    f".next_functions[{k}][0]"
                    for k in path_from_the_origin]
                saved_tensor_name = (
                    f"{main_target}.grad_fn" 
                    + "".join(path_str)
                    + "." + attr)
                saved_tensors.add((saved_tensor_name,attr_value))
        if hasattr(current_grad_fn,"next_functions"):
            for k,next_grad_fn in enumerate(current_grad_fn.next_functions):
                trace(next_grad_fn[0],path_from_the_origin+[k])
    trace(grad_fn,[])
    return explicit_vars,saved_tensors



def get_relevant_dependencies_via_grad_fn(
        sn_to_proceed : SimplifiedNode,
        our_global,tmp_local):
    # 1) init
    exec(
        sn_to_proceed.get_code(force_special_kwargs=True), 
        our_global, tmp_local)
    sn_value = tmp_local[sn_to_proceed.main_target]
    has_attribute__base = not (sn_value._base is None)

    # 2) Search through grad_fn
    (explicit_vars_in_grad_fn,
     saved_tensors_names_and_values) = trace_grad_fn(
        sn_value.grad_fn,
        sn_to_proceed.main_target,
        tmp_local["all_parameters"],
        tmp_local["all_input_values"]
    )

    # 3) find out which tensors we found are
    # We collected all tensors in grad_fn, and we need to find out
    # to which target they correspond to. For `saved_tensors`:
    # Case 1: we find a target in tmp_local with the same data_ptr: 
    # it confirms a dependency from sn to this target = real dependency
    # Case 2: It doesn't correspond to any dependency:
    # it means it's a fresh tensor only stored in grad_fn, 
    # what we call a "phantom"
    bwd_real_dependencies = set()
    saved_tensors_found = set()
    
    # To be more precise, for each dependency of sn 
    # we check whether it is present in grad_fn, 
    # i.e. whether it is a real dependency.
    potential_bwd_deps = sn_to_proceed.deps.union({sn_to_proceed})
    for req_sn in potential_bwd_deps:
        req_target = req_sn.main_target
        req_value = tmp_local[req_target]
        req_data_ptr = VariableInfo.get_data_ptr(req_value)
        found=False
        for explicit_var in explicit_vars_in_grad_fn:
            if req_data_ptr == VariableInfo.get_data_ptr(explicit_var):
                bwd_real_dependencies.add(req_sn.main_target)
                found=True
                break
        if not found:
            for saved_tensor in saved_tensors_names_and_values:
                saved_tensor_name,saved_tensor_value = saved_tensor
                if req_data_ptr == VariableInfo.get_data_ptr(saved_tensor_value):
                    bwd_real_dependencies.add(req_sn.main_target)
                    saved_tensors_found.append(saved_tensor_name)
                    saved_tensors_names_and_values.remove(saved_tensor)
                    break
    
    # 4) check whether the forward result data is a dependency
    if sn_to_proceed.main_target in bwd_real_dependencies:
        bool_bwd_requires_fwd_data = True
        bwd_real_dependencies.remove(sn_to_proceed.main_target)
    else:
        bool_bwd_requires_fwd_data = False

    # 5) check whether all saved_tensors were found
    bool_exist_phantoms = bool(saved_tensors_names_and_values != set())

    # 6) clean tmp_local, i.e. remove sn,
    # So we can reuse it to inspect time and memory usage
    for target in sn_to_proceed.all_targets:
        del tmp_local[target]

    return (bwd_real_dependencies,
        bool_bwd_requires_fwd_data,
        bool_exist_phantoms,
        has_attribute__base
    )
# ======================


# ======================



class inspector():
    # -> We define an inspector class to save every intermediate 
    # -> information used during inspection, very helpful to debug.
    def __init__(self,sn,sg,our_global,device):
        self.sn = sn
        self.sg = sg
        self.mt = sn.main_target
        self.info = sg.dict_info[self.mt]
        self.timer = measure.make_timer(device)
        self.memUsage = measure.MeasureMemory(device)
        self.our_global = our_global
        self.tmp_local = generate_tmp_local(sn,sg,our_global,device)
        self.ret = Inspection_result()
        self.ret.relevant = True
        self.device = device

    # ---------

    # === FORWARD ===
    # -- measure forward --
    def measure_fwd(self,only_run=False):
        def fct_run_fwd():
            self.code_run_fwd = self.sn.get_code(force_special_kwargs=True)
            exec(self.code_run_fwd, self.our_global, self.tmp_local)

        def fct_fgt_fwd():
            for tar in self.sn.tensor_targets:
                val = self.tmp_local[tar]
                val.data = torch.zeros(0,device=self.device)
                if val._base is not None:
                    val._base.data = torch.empty(0,device=self.device)
                
        def fct_del_fwd():
            code = ""
            for tar in self.sn.tensor_targets:
                code += f"del {tar};"
            self.code_del_fwd = code
            exec(self.code_del_fwd, self.our_global, self.tmp_local)

        gc.disable()
        # -> We don't want the gc to disturb the memory measurement
        _ , mem_run_fwd , peak_fwd = self.memUsage.measure(fct_run_fwd)
        overhead_fwd = peak_fwd - mem_run_fwd
        self.ret.overhead_fwd = overhead_fwd
        self.ret.mem_run_fwd = mem_run_fwd
        if not only_run:
            _ , mem_del_fwd , _ = self.memUsage.measure(fct_del_fwd)
            self.ret.mem_del_fwd = - mem_del_fwd
            fct_run_fwd()

            _ , mem_fgt_fwd , _ = self.memUsage.measure(fct_fgt_fwd)
            time_run_fwd = self.measure_time(fct_run_fwd)
            self.ret.mem_fgt_fwd = - mem_fgt_fwd
            self.ret.time_run_fwd = time_run_fwd
        gc.enable()
    # ===============

    # === BACKWARD ===

    def fct_run_bwd(self):
        self.code_run_bwd = f"{self.mt}.backward({self.mt}.grad)"
        exec(self.code_run_bwd, self.our_global, self.tmp_local)

    def fct_fgt_bwd(self):
        for req_sn in self.sn.deps.keys():
            if not req_sn.is_artifact:
                for tar in req_sn.tensor_targets:
                    self.tmp_local[tar].grad = None
    def fct_prepare_bwd(self):
        self.tmp_local = generate_tmp_local(
            self.sn,self.sg,self.our_global,self.device)
        exec(self.code_run_fwd, self.our_global, self.tmp_local)
        self.tmp_local[self.sn.main_target].grad = (
            self.info.generate_value(self.device))

    # measure
    def measure_bwd(self):
        if self.info.requires_grad:
            gc.disable() 
            self.fct_prepare_bwd()
            _ , mem_run_bwd , peak_bwd = (
                self.memUsage.measure(self.fct_run_bwd))
            overhead_bwd = peak_bwd - mem_run_bwd
            _ , mem_fgt_bwd , _ = self.memUsage.measure(self.fct_fgt_bwd)

            self.fct_prepare_bwd()
            time_run_bwd = self.measure_time(
                self.fct_run_bwd, self.fct_prepare_bwd)
            # overhead_bwd contains n.target.data now /!\
            gc.enable()
            self.ret.overhead_bwd = overhead_bwd
            self.ret.mem_run_bwd  = mem_run_bwd
            self.ret.mem_fgt_bwd  = - mem_fgt_bwd
            self.ret.time_run_bwd = time_run_bwd
    
            self.tmp_local.clear()

# ==========================

