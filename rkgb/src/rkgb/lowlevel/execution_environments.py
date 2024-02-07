import torch
from rkgb.lowlevel import ast_add_on
from rkgb.core import base


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


class EnvironmentGenerator():
    """
    To create global and local environments.
    - self.generate_local_env()
    - self.generate_global_env()
    """
    @staticmethod
    def generate_global_env(
            graph : base.Graph,
            current_device : torch.device,
            inspection_device : torch.device,
            original_mod = None):
        our_global = globals().copy()
        our_global["device"] = inspection_device
        for cst_name,cst_value in graph.dict_constants.items():
            our_global[cst_name] = cst_value.to(inspection_device)
        if current_device == inspection_device:
            assert original_mod is not None
            our_global["self"] = original_mod
        # Otherwise we create local FakeMod
        # that only have the required params
        return our_global
    



    @staticmethod
    def aux_generate_a_parameter_locally(
        param_node : base.ParameterNode,
        our_global, tmp_local,
        original_mod,
        current_device,
        inspection_device
    ):
        if current_device == inspection_device:
            # The whole model is already in our_global
            # So no FakeMod to create, but simply run view code
            param_value = param_node.get_value(original_mod)
            exec(param_node.get_code(),our_global,tmp_local)
        else:
            # Move to inspection device and FakeMod
            param_value = param_node.get_value(original_mod).to(inspection_device)
            if not param_node.is_buffer:
                param_value = torch.nn.Parameter(param_value,
                                                 requires_grad=param_node.get_value(original_mod).requires_grad)
            tmp_local["__value"] = param_value
            exec(f"{param_node.param_str} = __value ; {param_node.get_code()}",
                our_global, tmp_local)
        tmp_local["all_parameters_values"].append(param_value)
        tmp_local["all_parameters_strs"].append(param_node.param_str)


    @staticmethod
    def _init_tmp_local(current_device,inspection_device):
        tmp_local = dict()
        if current_device != inspection_device:
            tmp_local["self"] = FakeMod()
        tmp_local["all_parameters_strs"] = [] # to find them easily
        tmp_local["all_parameters_values"] = []
        tmp_local["all_inputs_values"] = set()
        return tmp_local



    @staticmethod
    def generate_local_env_with_forward(
            fn_to_proceed : base.Node,
            forward_graph : base.Graph,
            our_global : dict,
            original_mod : torch.nn.Module,
            current_device : torch.device,
            inspection_device : torch.device):
        assert type(fn_to_proceed).__name__ == "ForwardNode"
        assert type(forward_graph).__name__ == "ForwardGraph"
        tmp_local = EnvironmentGenerator._init_tmp_local(current_device,inspection_device)
        # 0) Generate required parameters
        for param_node in fn_to_proceed.required_parameter_nodes:
            EnvironmentGenerator.aux_generate_a_parameter_locally(
                param_node,our_global,tmp_local,original_mod,
                current_device,inspection_device)

        # To generate an environment where to run raw_node's code,
        # We generate fn's dependencies, either using their info 
        # (type, shape, dtype etc) we previously collected, 
        # or by running their code in case of view or inplace nodes, 
        # in which case we first (i) generate their dependencies, 
        # using previously collected info; and (ii) its random dependencies.
        targets_done = set()
        targets_ready = set()
        nodes_todo = list(fn_to_proceed.deps)
        while nodes_todo != []:
            req_fn = nodes_todo[-1]
            req_target = req_fn.target
            if req_target in targets_done or req_target in our_global:
                nodes_todo.pop()
            else:
                req_fn_info = forward_graph.dict_info[req_target]
                if (req_fn_info.is_inplace 
                or  req_fn_info.is_view
                or  req_fn.fct == "getattr"):
                    # For a view: we first generate its deps
                    if req_target in targets_ready:
                        # When all deps are done, then we run the viewing code
                        # Don't forgot : param deps => in case of view over a param
                        for param_node in req_fn.required_parameter_nodes:
                            EnvironmentGenerator.aux_generate_a_parameter_locally(
                                param_node,our_global,tmp_local,original_mod,
                                current_device,inspection_device)
                        # And required random stuff
                        for req_rd in req_fn.required_random_tensors:
                            if not req_rd in targets_done:
                                code = ast_add_on.make_str_assign(
                                    (req_rd,forward_graph.dict_rand[req_rd]))
                                exec(code,our_global,tmp_local)
                                targets_done.add(req_rd)
                        # Exec the req_fn:
                        exec(req_fn.get_code(),our_global,tmp_local)
                        targets_done.add(req_target)
                        nodes_todo.pop()
                    else:
                        nodes_todo.extend(list(req_fn.deps))
                        targets_ready.add(req_target)
                else:
                    req_x = req_fn_info.generate_value(inspection_device)
                    if isinstance(req_x,torch.Tensor):
                        req_x = req_x.clone()
                    tmp_local[req_target] = req_x
                    targets_done.add(req_target)
                    nodes_todo.pop()
        return tmp_local
    




    @staticmethod
    def generate_local_env_with_simplified(
            sn_to_proceed : base.Node,
            simplified_graph : base.Graph,
            our_global : dict,
            original_mod : torch.nn.Module,
            current_device : torch.device,
            inspection_device : torch.device):
        assert type(sn_to_proceed).__name__ == "SimplifiedNode"
        assert type(simplified_graph).__name__ == "SimplifiedGraph"
        tmp_local = EnvironmentGenerator._init_tmp_local(current_device,inspection_device)
        all_inputs = (
            simplified_graph.original_mod_input_targets
            + simplified_graph.input_targets)
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
                EnvironmentGenerator.aux_generate_a_parameter_locally(
                    param_node,our_global,tmp_local,original_mod,
                    current_device,inspection_device)
            exec(
                init_node.get_code(force_special_kwargs=True),
                our_global,tmp_local)
            for inp in all_inputs:
                tmp_local["all_inputs_values"].add(tmp_local[inp])
        else:
            # We don't need any tensor:
            # we generate sizes anyway as they come free 
            for inp in all_inputs:
                inp_info = simplified_graph.dict_info[inp]
                if inp_info.variable_type is not torch.Tensor:
                    inp_value = inp_info.generate_value(inspection_device)
                    tmp_local[inp] = inp_value
                    tmp_local["all_inputs_values"].add(inp_value)

        # 2) Generate required parameters
        for param_node in sn_to_proceed.required_parameter_nodes:
            EnvironmentGenerator.aux_generate_a_parameter_locally(
                param_node,our_global,tmp_local,original_mod,
                current_device,inspection_device)

        # 3) Generate all the deps
        list_nodes_to_generate = list(sn_to_proceed.deps)
        set_nodes_to_generate = set(list_nodes_to_generate)
        while list_nodes_to_generate != []:
            # Get next node to generate
            sn_to_generate = list_nodes_to_generate.pop(0)
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
                    EnvironmentGenerator.aux_generate_a_parameter_locally(
                        req_param_node,our_global,tmp_local,original_mod,
                        current_device,inspection_device)
                for req_var_target in simplified_graph.dict_target_to_direct_variable_deps[body_target]:
                    req_var_info = simplified_graph.dict_info[req_var_target]
                    tmp_local[req_var_target] = req_var_info.generate_value(inspection_device)
            exec(body_code,our_global,tmp_local)
        return tmp_local




    @staticmethod
    def generate_local_env(
            node_to_proceed : base.Node,
            graph : base.Graph,
            our_global : dict,
            original_mod : torch.nn.Module,
            current_device : torch.device,
            inspection_device : torch.device):
        if (type(node_to_proceed).__name__ == "SimplifiedNode"
        and type(graph).__name__ == "SimplifiedGraph"):
            method = EnvironmentGenerator.generate_local_env_with_simplified
        else:
            method = EnvironmentGenerator.generate_local_env_with_forward
        return method(
                node_to_proceed,graph,our_global,original_mod,
                current_device,inspection_device
            )
    
