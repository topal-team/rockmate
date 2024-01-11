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
        self.time_run_fwd = 0
        self.time_run_bwd = 0


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
        param_value = param_node.get_value(original_mod).to(inspection_device)
        tmp_local["all_parameters"].add(param_value)
        tmp_local["__value"] = param_value
        exec(f"{param_node.param_str} = __value ; {param_node.get_code()}",
            our_global, tmp_local)

    @staticmethod
    def generate_local_env(
            simplified_node_for_whom_to_generate_env : SimplifiedNode,
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
        tmp_local["all_input_targets"] = all_inputs
        tmp_local["all_input_values"] = set()
        # 1) Do we need to run the init_code:
        # - Generating the sizes related to init_code is free
        # so we can do it anyway, but if we require a tensor
        # for the moment I generate all the real inputs and then
        # run the init_code, as the tensor we need may be a view
        # TO IMPROVE ? Generate exactly the tensors needed
        init_node = simplified_graph.init_node
        if (
                ((simplified_node_for_whom_to_generate_env,init_node)
                in simplified_graph.dict_of_labels_on_edges)
        and 
            any(
                simplified_graph.dict_info[needed_input].variable_type 
                is torch.Tensor
                for needed_input 
                in simplified_graph.dict_of_labels_on_edges[
                    (simplified_node_for_whom_to_generate_env,init_node)
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
        for param_node in simplified_node_for_whom_to_generate_env.required_parameter_nodes:
            Inspector.aux_generate_a_parameter_locally(
                param_node,our_global,tmp_local,
                original_mod,inspection_device)

        # 3) Generate all the deps
        list_nodes_to_generate = list(simplified_node_for_whom_to_generate_env.deps)
        set_nodes_to_generate = set(list_nodes_to_generate)
        while list_nodes_to_generate != []:
            # Get next node to generate
            sn : SimplifiedNode = list_nodes_to_generate.pop(0)
            if sn is init_node: # TO REMOVE
                raise Exception("init_node in sn.deps ???")
            # Check if it's `sn`'s turn:
            # if some of the deps of sn are in the waiting list
            # ie we plan to properly generate them (because they
            # are also in main_sn's deps) it's better to wait.
            # But note that we don't add any additional node to 
            # the waiting list. So latter on, for sn'deps which
            # aren't main_sn'deps: we will just generate them on the fly.
            if set(sn.deps).intersection(set_nodes_to_generate) != set():
                list_nodes_to_generate.append(sn) # not his turn yet
                continue
            else:
                set_nodes_to_generate.remove(sn)

            # We are ready to generate sn:
            # - First we create the main_target value based on info
            # - Then we run the body_code to generate views / sizes
            main_value = sn.info.generate_value(inspection_device)
            # Some operations are impossible over leaf tensors 
            # in term of grad_fn. So we have to clone them :
            if isinstance(main_value,torch.Tensor):
                main_value = main_value.clone()
            tmp_local[sn.main_target] = main_value
            
            # To run the body code we may need some dependencies to be
            # in tmp_local (e.g. sizes): so we create them on the fly
            # Note: a dependency of sn which is also used by
            # simplified_node_for_whom_to_generate_env isn't created 
            # from info but previously generated in this while loop
            body_code = ast_add_on.make_str_list_assign(
                sn.body_code, force_special_kwargs=True)
            for body_target in sn.all_targets:
                if body_target is sn.main_target: continue
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
# TRACE GRAD_FN TO KNOW
# WHAT IS NEEDED TO BWD
# -> REAL_DEPS
# -> PHANTOMS
# ======================

# -> auxiliary function for "get_useful_vars" below
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
        simplified_node : SimplifiedNode,
        our_global,tmp_local):
    # 1) init
    exec(
        simplified_node.get_code(force_special_kwargs=True), 
        our_global, tmp_local)
    sn_value = tmp_local[simplified_node.main_target]
    has_attribute__base = not (sn_value._base is None)

    # 2) Search through grad_fn
    (explicit_vars_in_grad_fn,
     saved_tensors_names_and_values) = trace_grad_fn(
        sn_value.grad_fn,
        simplified_node.main_target,
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
    real_dependencies_of_sn = set()
    saved_tensors_found = set()
    
    # To be more precise, for each dependency of sn 
    # we check whether it is present in grad_fn, 
    # i.e. whether it is a real dependency.
    for req_sn in simplified_node.deps:
        req_target = req_sn.main_target
        req_value = tmp_local[req_target]
        req_data_ptr = VariableInfo.get_data_ptr(req_value)
        found=False
        for explicit_var in explicit_vars_in_grad_fn:
            if req_data_ptr == VariableInfo.get_data_ptr(explicit_var):
                real_dependencies_of_sn.add(req_sn)
                found=True
                break
        if not found:
            for saved_tensor in saved_tensors_names_and_values:
                saved_tensor_name,saved_tensor_value = saved_tensor
                if req_data_ptr == VariableInfo.get_data_ptr(saved_tensor_value):
                    real_dependencies_of_sn.add(req_sn)
                    saved_tensors_found.append(saved_tensor_name)
                    saved_tensors_names_and_values.remove(saved_tensor)
                    break
    
    # 4) check whether all saved_tensors were found
    exist_phantoms = bool(saved_tensors_names_and_values != set())

    # 5) clean tmp_local, i.e. remove sn,
    # So we can reuse it to inspect time and memory usage
    for target in simplified_node.all_targets:
        del tmp_local["target"]

    return (real_dependencies_of_sn,
        exist_phantoms,
        has_attribute__base
    )

# ======================



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

    # -- aux --
    def measure_time(self, fct, inter_fct=None):
        t = self.timer.measure(fct)
        nb_repeat = 1
        measures = [t] ; tot = t
        while (tot < constants.time_min_duration
        or nb_repeat < constants.time_min_repeat):
            if inter_fct: inter_fct()
            t = self.timer.measure(fct)
            measures.append(t)
            tot += t ; nb_repeat += 1
        if len(measures)>2:
            return (
                (sum(measures)-max(measures)-min(measures))
                /(len(measures)-2))
        else:np.median(measures)
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

