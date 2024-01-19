from rkgb.lowlevel import preprocess_device
from rkgb.lowlevel import preprocess_samples
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
            ):
        self.original_mod = model
        self.inspection_device = inspection_device
        self.use_jit_instead_of_dynamo = use_jit_instead_of_dynamo
        self.jit_impose_device = jit_impose_device
        self.partitioners = partitioners

        # 0) See what we want
        bool_h = "H" in wanted_graphs
        bool_p = ("P" in wanted_graphs) or bool_h
        bool_fb = ("FB" in wanted_graphs) or bool_h
        bool_s = ("S" in wanted_graphs) or bool_fb
        bool_f = ("F" in wanted_graphs) or bool_s
        bool_r = ("R" in wanted_graphs) or bool_f
        
        # 1) device and inputs
        self.example_inputs = preprocess_samples.ExampleInputs(model,model_args,model_kwargs)
        self.current_device = preprocess_device.get_device_and_check_all_same_device(model,self.example_inputs)

        # 2) Build everything
        if bool_r: self.build_raw()
        if bool_f: self.build_forward()
        if bool_s: self.build_simplified()
        if bool_fb: self.build_forward_and_backward()
        if bool_p: self.build_partitioned()
        if bool_h: self.build_hierarchical()

    def build_raw(self):
        if self.raw_graph is None:
            self.raw_graph = RawGraph(
                self.original_mod,
                self.example_inputs,
                use_jit_instead_of_dynamo=self.use_jit_instead_of_dynamo,
                jit_impose_device=self.jit_impose_device)
            
    def build_forward(self):
        if self.forward_graph is None:
            self.build_raw()
            self.forward_graph = ForwardGraph(
                self.raw_graph,
                self.original_mod,
                self.example_inputs,
                self.current_device)
            
    def build_simplified(self):
        if self.simplified_graph is None:
            self.build_forward()
            self.simplified_graph = SimplifiedGraph(
                self.forward_graph,
                self.original_mod,
                self.current_device)

    def build_forward_and_backward(self):
        if self.forward_and_backward_graph is None:
            self.build_simplified()
            if self.inspection_device is None:
                self.inspection_device = self.current_device
            self.forward_and_backward_graph = ForwardAndBackwardGraph(
                self.simplified_graph,
                self.original_mod,
                self.inspection_device)

    def build_partitioned(self,partitioners = None):
        partitioners = partitioners or self.partitioners
        if self.partitioned_structure is None or partitioners is not None:
            self.build_simplified()
            if partitioners is None:
                partitioners = [partitioned.PartitionerBottomToTop()]
            self.partitioned_structure = partitioned.PartitionedStructure(
                self.simplified_graph,
                self.original_mod,
                partitioners)
            
    def build_hierarchical(self,partitioners = None):
        if self.hierarchical_structure is None or partitioners is not None:
            self.build_forward_and_backward()
            self.build_partitioned(partitioners)
            self.hierarchical_structure = HierarchicalStructure(
                self.partitioned_structure,
                self.forward_and_backward_graph)

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

