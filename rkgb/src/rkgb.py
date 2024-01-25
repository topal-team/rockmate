

import time
pip_editable_broken_imports = False
if pip_editable_broken_imports:
    from lowlevel import preprocess_device
    from lowlevel import preprocess_samples
    from lowlevel import anonymize
    from core.raw import RawGraph
    from core.forward import ForwardGraph
    from core.simplified import SimplifiedGraph
    from core.backward import ForwardAndBackwardGraph
    from core import partitioned
    from core.hierarchical import HierarchicalStructure
else:
    from rkgb.lowlevel import preprocess_device
    from rkgb.lowlevel import preprocess_samples
    from rkgb.lowlevel import anonymize
    from rkgb.core.raw import RawGraph
    from rkgb.core.forward import ForwardGraph
    from rkgb.core.simplified import SimplifiedGraph
    from rkgb.core.backward import ForwardAndBackwardGraph
    from rkgb.core import partitioned
    from rkgb.core.hierarchical import HierarchicalStructure


class Result():
    raw_graph = None
    forward_graph = None
    simplified_graph = None
    dict_mt_to_sn_ano_material = None
    forward_and_backward_graph = None
    partitioned_structure = None
    hierarchical_structure = None
    def __init__(self,
            model,model_args,model_kwargs=None,
            wanted_graphs = {"R","F","S","FB","P","H"},
            inspection_device = None,
            use_jit_instead_of_dynamo = False,
            jit_impose_device = True,
            partitioners = None,
            print_time_in_each_stage = False,
            do_inspection = True,
            ):
        self.original_mod = model
        self.inspection_device = inspection_device
        self.use_jit_instead_of_dynamo = use_jit_instead_of_dynamo
        self.jit_impose_device = jit_impose_device
        self.partitioners = partitioners
        self.last_partitioners = None
        self.print_time_in_each_stage = print_time_in_each_stage
        self.do_inspection = do_inspection
        self.last_time = 0

        # 1) device and inputs
        self.example_inputs = preprocess_samples.ExampleInputs(model,model_args,model_kwargs)
        self.current_device = preprocess_device.get_device_and_check_all_same_device(model,self.example_inputs)

        # 2) Build everything
        if "R" in wanted_graphs: self.build_raw()
        if "F" in wanted_graphs: self.build_forward()
        if "S" in wanted_graphs: self.build_simplified()
        if "P" in wanted_graphs: self.build_partitioned()
        if "FB" in wanted_graphs: self.build_forward_and_backward()
        if "H" in wanted_graphs: self.build_hierarchical()

    def start_time(self):
        self.last_time = time.time()
    def show_time(self,stage):
        if self.print_time_in_each_stage:
            time_taken = time.time() - self.last_time
            clean_time_taken = time.strftime("%H:%M:%S", time.gmtime(time_taken))
            print(f"Stage {stage} took {clean_time_taken}")

    def build_raw(self):
        if self.raw_graph is None:
            self.start_time()
            self.raw_graph = RawGraph(
                self.original_mod,
                self.example_inputs,
                use_jit_instead_of_dynamo=self.use_jit_instead_of_dynamo,
                jit_impose_device=self.jit_impose_device)
            self.show_time("Raw")
            
    def build_forward(self):
        if self.forward_graph is None:
            self.build_raw()
            if self.inspection_device is None:
                self.inspection_device = self.current_device
            self.start_time()
            self.forward_graph = ForwardGraph(
                self.raw_graph,
                self.original_mod,
                self.example_inputs,
                self.current_device,
                self.inspection_device)
            self.show_time("Forward")
            
    def build_simplified(self):
        if self.simplified_graph is None:
            self.build_forward()
            self.start_time()
            self.simplified_graph = SimplifiedGraph(
                self.forward_graph,
                self.original_mod,
                self.current_device)
            self.show_time("Simplifications")

    def build_anonymization(self):
        if self.dict_mt_to_sn_ano_material is None:
            self.build_simplified()
            (self.dict_target_ano_id,
            self.dict_mt_to_sn_ano_material ) \
                = anonymize.build_anonymous_equivalence_classes(
                    self.simplified_graph,self.original_mod)


    def build_partitioned(self,partitioners = None):
        partitioners = partitioners or self.partitioners
        if (self.partitioned_structure is None 
        or partitioners != self.last_partitioners):
            self.build_simplified()
            self.build_anonymization()
            self.start_time()
            if partitioners is None:
                partitioners = [partitioned.default_partitioner]
            self.partitioned_structure = partitioned.PartitionedStructure(
                self.simplified_graph,
                partitioners,
                self.dict_target_ano_id,
                self.dict_mt_to_sn_ano_material)
            self.last_partitioners = partitioners
            self.show_time("Partitioning")
            
    def build_forward_and_backward(self):
        if self.forward_and_backward_graph is None:
            self.build_simplified()
            self.build_anonymization()
            if self.inspection_device is None:
                self.inspection_device = self.current_device
            self.start_time()
            self.forward_and_backward_graph = ForwardAndBackwardGraph(
                self.simplified_graph,
                self.original_mod,
                self.current_device,
                self.inspection_device,
                self.do_inspection,
                self.dict_mt_to_sn_ano_material)
            self.show_time("ForwardAndBackward")

    def build_hierarchical(self,partitioners = None):
        if self.hierarchical_structure is None or partitioners is not None:
            self.build_partitioned(partitioners)
            self.build_forward_and_backward()
            self.start_time()
            self.hierarchical_structure = HierarchicalStructure(
                self.partitioned_structure,
                self.forward_and_backward_graph)
            self.hierarchical_cluster = self.hierarchical_structure.main_cluster
            self.show_time("Hierarchical")

    @property
    def R(self):
        return self.raw_graph
    @property
    def F(self):
        return self.forward_graph
    @property
    def S(self):
        return self.simplified_graph
    @property
    def FB(self):
        return self.forward_and_backward_graph
    @property
    def Ps(self):
        return self.partitioned_structure
    @property
    def Pc(self):
        if hasattr(self.partitioned_structure,"main_cluster"):
            return self.partitioned_structure.main_cluster
        else:
            return None
    @property
    def Hs(self):
        return self.hierarchical_structure
    @property
    def Hc(self):
        if hasattr(self.hierarchical_structure,"main_cluster"):
            return self.hierarchical_structure.main_cluster
        else:
            return None

# ==========================

