from rkgb.lowlevel.ast_add_on import make_str_assign, make_str_list_assign
from rkgb.core.backward import ComputationNode, AllocationNode
import torch
import numpy as np
from .op_schedule import (
    Allocation,
    Activation,
    Parameter,
    Buffer,
    Op,
    ComputeOp,
    DeleteOp,
    MappingOp,
    AllocateOp,
    OffloadOp,
    PrefetchOp,
    SynchronizeOp,
    OptimizeOp,
    OpSchedule,
    ExecCodeOp,
    PrepareOp,
)
import psutil


class RngState:
    """
    We take care of random operations,
    in particular to be able to recompute a random function deterministically,
    we store random states on the first computation and restore them when needed.
    """

    def __init__(self):
        self.cpu_states = {}
        self.gpu_states = {}

    def get(self, op_name):
        if op_name not in self.cpu_states.keys():
            self.cpu_states[op_name] = torch.get_rng_state()
            self.gpu_states[op_name] = torch.cuda.get_rng_state()

    def restore(self, op_name):
        # pass
        torch.set_rng_state(self.cpu_states[op_name])
        torch.cuda.set_rng_state(self.gpu_states[op_name])


def make_gd(
    device,
    nn_mod,
    dict_constants,
    # cpu_optim,
    # gpu_optim,
    # optim_kwargs={},
    optimize_metrics={},
):
    return {
        **globals(),
        **dict_constants,
        # "original_mod": nn_mod,
        "self": nn_mod,
        "device": device,
        "torch": torch,
        "meta": torch.ones(1).to(device),
        "cmeta": torch.view_as_complex(torch.ones(2)).to(device),
        "cpu_optim": optimize_metrics["cpu_optim"],
        "gpu_optim": optimize_metrics["gpu_optim"],
        "opt_kwargs": optimize_metrics["optim_kwargs"],
        "optimize_metrics": optimize_metrics,
        "main_stream": torch.cuda.current_stream(),
        # "prefetch_stream": torch.cuda.current_stream(),
        # "offload_stream": torch.cuda.current_stream(),
        "prefetch_stream": torch.cuda.Stream(device),
        "offload_stream": torch.cuda.Stream(device),
    }


class RK_Storage:
    """
    There can only be one RK_Storage.
    All the changes will be made correspond to it.
    All the RK_Fct share the storage.
    """

    _instance = None

    def __init__(self):
        pass

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def init(self, gd):
        self.ld = dict()
        self.shapes = {"Empty_size": torch.Size([0])}
        self.dtypes = dict()
        self.rng_state = RngState()
        self.gd = gd

    def add_val(self, val, x):
        self.ld[val] = x


class RK_Fct:
    def __init__(self, target_name, storage, **kwargs):
        self.storage = storage
        self.target_name = target_name
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.target_name}"


class Fct_del(RK_Fct):
    def __init__(self, target_name: str, storage: RK_Storage, del_mode="data"):
        super().__init__(target_name=target_name, storage=storage)
        self.del_fcts = {
            "data": self.del_data,
            "base": self.del_base,
            "grad": self.del_grad,
            "var": self.del_var,
            "optim_states": self.del_optim_states,
        }
        self.del_mode = del_mode

    def __call__(self):
        self.del_fcts[self.del_mode]()

    def del_data(self):
        self.storage.ld[self.target_name].data = torch.empty(0)
        # pass

    def del_grad(self):
        self.storage.ld[self.target_name].grad = None

    def del_base(self):
        if self.storage.ld[self.target_name]._base is None:
            return
        self.storage.ld[self.target_name]._base.data = torch.empty(0)

    def del_var(self):
        self.storage.ld[self.target_name] = torch.empty(0)

    def del_optim_states(self):
        for k, v in self.storage.ld[f"Optimize_({self.target_name})"].state.items():
            v["exp_avg"].data = torch.empty(0)
            v["exp_avg_sq"].data = torch.empty(0)


class Fct_gen_fake_data(RK_Fct):
    def __init__(
        self, target_name: str, storage: RK_Storage, with_proxy=False, **kwargs
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.with_proxy = with_proxy

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            m = (
                self.storage.gd["cmeta"]
                if self.storage.dtypes[self.target_name].is_complex
                else self.storage.gd["meta"]
            )
            s = self.storage.shapes[self.target_name]
            if s == torch.Size([]):
                x = m.sum()  # easy way to obtain a Tensor of shape []
            else:
                x = m.expand(np.prod(s)).view(s)
            self.storage.ld[self.target_name].data = x
            if self.with_proxy:
                self.storage.ld[f"_{self.target_name}"].data = x


class Fct_detach(RK_Fct):
    def __init__(self, target_name: str, storage: RK_Storage, **kwargs):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name

    def fake_detach(self):
        """
        For certain in-place operations, detach could lead to wrong memory usage.
        Fake detach is to solve it by merging the operations.
        TODO: test on ResNet
        """
        self.storage.ld[self.target_name] = self.storage.ld[f"_{self.target_name}"]
        self.storage.ld[f"_{self.target_name}"] = torch.empty(0)

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.storage.ld[self.target_name].data = self.storage.ld[
                f"_{self.target_name}"
            ].data


class Fct_run_bwd(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        retain_graph=False,
        input_names=[],
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.retain_graph = retain_graph
        self.input_names = input_names

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            inputs = [self.storage.ld[name] for name in self.input_names]
            if not inputs:
                inputs = None
            self.storage.ld[f"_{self.target_name}"].backward(
                self.storage.ld[self.target_name].grad,
                inputs=inputs,
                retain_graph=self.retain_graph,
            )


class Fct_run_fwd(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        code,
        no_save_list=[],
        fwd_mode="with_grad",
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.code = code
        self.no_save_list = no_save_list
        self.fwd_fct = {"with_grad": self.fwd_with_grad, "no_grad": self.fwd_no_grad}
        self.fwd_mode = fwd_mode

    def fwd_with_grad(self):
        with torch.enable_grad():
            with torch.autograd.graph.saved_tensors_hooks(
                self.fct_get_pack(self.no_save_list), self.fct_get_unpack()
            ):
                exec(self.code, self.storage.gd, self.storage.ld)

    def fwd_no_grad(self):
        with torch.no_grad():
            exec(self.code, self.storage.gd, self.storage.ld)

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.fwd_fct[self.fwd_mode]()

    def fct_get_pack(self, no_save_list, sanity_check=False):
        # no_save_list contains a list of names
        def pack(x):
            for i, c in enumerate(no_save_list):
                if self.storage.ld[c].data_ptr() == x.data_ptr():
                    # print(c)
                    if sanity_check:
                        assert torch.equal(
                            self.storage.ld[c].data.as_strided_(
                                x.shape, x.stride(), x.storage_offset()
                            ),
                            x,
                        )
                    return (
                        c,
                        x.shape,
                        x.stride(),
                        x.storage_offset(),
                        # x.clone(),
                    )
            return x

        return pack

    def fct_get_unpack(self):
        def unpack(x):
            if isinstance(x, tuple):
                target = self.storage.ld[x[0]]
                return target.as_strided(*x[1:4])
            return x

        return unpack


class Fct_get_shape(RK_Fct):
    def __init__(self, target_name: str, storage: RK_Storage, pnode=None, **kwargs):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.pnode = pnode

    def __call__(self):
        if self.pnode:
            target = self.pnode.get_value(self.storage.gd["self"])
        else:
            target = self.storage.ld[self.target_name]

        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.storage.shapes[self.target_name] = target.shape
            self.storage.shapes[f"cpu_{self.target_name}"] = target.shape
            self.storage.dtypes[self.target_name] = target.dtype
            self.storage.dtypes[f"cpu_{self.target_name}"] = target.dtype


class Fct_optimize(RK_Fct):
    def __init__(
        self, target_name: str, storage: RK_Storage, del_grad_list: list = [], **kwargs
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.del_grad_list = del_grad_list

    def __call__(self):
        # if "cpu" in self.target_name:
        #     torch.cuda.synchronize()
            # return None
        self.storage.ld[self.target_name].step()
        # for p in self.del_grad_list:
        #     self.storage.ld[p].grad.zero_()


class Fct_mem_alloc(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        shape=None,
        dtype=None,
        alloc_mode="data",
        device=None,
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        # self.target = target
        self.target_name = target_name
        self.alloc_fct = {
            "data": self.alloc_data,
            "grad": self.alloc_grad,
            "tensor": self.alloc_tensor,
            "optim_states": self.alloc_optim_states,
        }
        self.alloc_mode = alloc_mode
        if device is None:
            self.device = self.storage.gd["device"]
        else:
            self.device = device
        self.kwargs = kwargs
        if self.device == torch.device("cpu"):
            self.kwargs["pin_memory"] = True
        self.shape_name = target_name if shape is None else shape

        self.dtype = dtype

    def alloc_optim_states(self):
        for k, v in self.storage.ld[f"Optimize_({self.target_name})"].state.items():
            v["exp_avg"].data = torch.empty_like(
                self.storage.ld[f"exp_avg_{self.target_name}"], device=self.device
            )
            v["exp_avg_sq"].data = torch.empty_like(
                self.storage.ld[f"exp_avg_sq_{self.target_name}"], device=self.device
            )

    def alloc_grad(self):
        shape = self.storage.shapes[self.shape_name]
        dtype = (
            self.storage.dtypes[self.shape_name] if self.dtype is None else self.dtype
        )
        self.storage.ld[self.target_name].grad = torch.empty(
            shape, dtype=dtype, device=self.device, **self.kwargs
        )

    def alloc_data(self):
        shape = self.storage.shapes[self.shape_name]
        dtype = (
            self.storage.dtypes[self.shape_name] if self.dtype is None else self.dtype
        )
        self.storage.ld[self.target_name].data = torch.empty(
            shape, dtype=dtype, device=self.device, **self.kwargs
        )

    def alloc_tensor(self):
        shape = self.storage.shapes[self.shape_name]
        dtype = (
            self.storage.dtypes[self.shape_name] if self.dtype is None else self.dtype
        )
        self.storage.ld[self.target_name] = torch.empty(
            shape, dtype=dtype, device=self.device, **self.kwargs
        )

    def __call__(self):
        with torch.cuda.stream(self.storage.gd["main_stream"]):
            self.alloc_fct[self.alloc_mode]()


class Fct_offload(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        offload_mode="param",
        stream="offload_stream",
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.offload_fct = {
            "param": self.offload_param,
            "grad": self.offload_grad,
            "optim_states": self.offload_optim_states,
        }
        self.offload_mode = offload_mode
        self.stream = stream

    def offload_param(self):
        self.storage.ld[f"cpu_{self.target_name}"].data.copy_(
            self.storage.ld[self.target_name].data,
            non_blocking=True,
        )

    def offload_grad(self):
        self.storage.ld[f"cpu_{self.target_name}"].grad.data.copy_(
            self.storage.ld[self.target_name].grad,
            non_blocking=True,
        )

    def offload_optim_states(self):
        for k, v in self.storage.ld[f"Optimize_({self.target_name})"].state.items():
            self.storage.ld[f"exp_avg_{self.target_name}"].copy_(
                v["exp_avg"], 
                non_blocking=True
            )
            self.storage.ld[f"exp_avg_sq_{self.target_name}"].copy_(
                v["exp_avg_sq"], 
                non_blocking=True
            )

    def __call__(self):
        with torch.cuda.stream(self.storage.gd[self.stream]):
            pass
            self.offload_fct[self.offload_mode]()


class Fct_prefetch(RK_Fct):
    def __init__(
        self,
        target: Allocation,
        storage: RK_Storage,
        prefetch_mode="param",
        stream="prefetch_stream",
        **kwargs,
    ):
        super().__init__(target_name=target.name, storage=storage)
        self.target = target
        self.target_name = target.target_name
        self.prefetch_fct = {
            "param": self.prefetch_param,
            "optim_states": self.prefetch_optim_states,
        }
        self.post_process = {
            "param": self.post_process_param,
            "optim_states": self.post_process_optim_states,
        }
        self.prefetch_mode = prefetch_mode
        self.stream = stream
        self.post_process_code = ""
        if isinstance(target, Parameter):
            self.post_process_code = target.pnode.get_code()

    def prefetch_param(self):
        self.storage.ld[self.target_name].data.copy_(
            self.storage.ld[f"cpu_{self.target_name}"].data,
            non_blocking=True,
        )

    # def prefetch_grad(self):
    #     self.storage.ld[f"cpu_{self.target_name}"].grad.data.copy_(
    #                 self.storage.ld[self.target_name].grad,
    #                 non_blocking=True,
    #             )
    def prefetch_optim_states(self):
        for k, v in self.storage.ld[f"Optimize_({self.target_name})"].state.items():
            v["exp_avg"].copy_(
                self.storage.ld[f"exp_avg_{self.target_name}"], non_blocking=True
            )
            v["exp_avg_sq"].copy_(
                self.storage.ld[f"exp_avg_sq_{self.target_name}"], non_blocking=True
            )

    def post_process_param(self):
        pass
        # with torch.enable_grad():
        #     exec(self.post_process_code, self.storage.gd, self.storage.ld)

    def post_process_optim_states(self):
        pass

    def __call__(self):
        with torch.cuda.stream(self.storage.gd[self.stream]):
            self.prefetch_fct[self.prefetch_mode]()
            # torch.cuda.synchronize()
            # with torch.cuda.stream(self.storage.gd["main_stream"]):
            # self.post_process[self.prefetch_mode]()


class Fct_synchronize(RK_Fct):
    def __init__(self, storage: RK_Storage, stream=None, **kwargs):
        super().__init__(target_name=f"Synchronize_{stream}", storage=storage)
        self.stream = stream
        self.target_name = f"Synchronize_{stream}"

    def wait_stream(self):
        torch.cuda.stream(self.storage.gd[self.stream]).synchronize()

    def sync_all(self):
        torch.cuda.synchronize()

    def __call__(self):
        if self.stream:
            self.wait_stream()
        else:
            self.sync_all()


class Fct_RNG_state(RK_Fct):
    def __init__(self, target_name: str, storage: RK_Storage, get_state=True, **kwargs):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.get_state = get_state

    def __call__(self):
        if self.get_state:
            self.storage.rng_state.get(self.target_name)
        else:
            self.storage.rng_state.restore(self.target_name)


class Fct_to_storage(RK_Fct):
    """
    Get parameter value from nn.module (storage.ld['self']),
    send to the value in storage.ld[target_name].
    By default, module parameters will be kept in CPU (f'cpu_{target_name}')
    and the real value should be empty.
    """

    def __init__(self, target_name, storage: RK_Storage, pnode, device=None, **kwargs):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.pnode = pnode
        self.device = device

    def __call__(self):
        self.storage.ld[self.pnode.param_name] = self.pnode.get_value(
            self.storage.gd["self"]
        )
        self.storage.ld[self.target_name].data = self.storage.ld[
            self.pnode.param_name
        ].data.to(self.device)


class Fct_add_optimizer(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        list_params,
        optim,
        is_cpu=False,
        **kwargs,
    ):
        super().__init__(
            target_name=target_name,
            storage=storage,
        )
        self.target_name = target_name
        self.list_params = list_params
        self.optim = optim
        self.kwargs = kwargs
        self.is_cpu = is_cpu

    def __call__(self):
        self.storage.ld[self.target_name] = self.optim(
            [self.storage.ld[self.is_cpu * "cpu_" + p] for p in self.list_params],
            **self.kwargs,
        )


class Compiler:
    """
    New structure:
    RK_storage will contain ld and gd, ld is for the
    temporary storage like activations and tensors generated
    for execution (including the views of params).
    If not parameter offloading, we assume the activations
    should be released by the end of each iteration.
    Therefore ld.clear() can be called after each iteration.
    The parameters should be returned back to the nn.module.
    Optimizer states are generated during the iteration,
    but they are used for future potential training thus
    should be allowed to be kept after iterations.
    By default, it should be released during reinit process.
    """

    def __init__(self, storage):
        self.storage = storage

        self.compile_op = {
            "ComputeOp": self.Compute,
            "DeleteOp": self.Delete,
            "OffloadOp": self.Offload,
            "PrefetchOp": self.Prefetch,
            "AllocateOp": self.Allocate,
            "OptimizeOp": self.Optimize,
            "SynchronizeOp": self.Synchronize,
            "PrepareOp": self.Prepare,
            "Op": lambda x: None,  # only for preparation
        }

    def compile_sched(self, op_sched: OpSchedule):
        op_sched.add_pos_info()
        for i, op in enumerate(op_sched.init_op_list):
            self.compile_op[op.__class__.__name__](op)
        for i, op in enumerate(op_sched.op_list):
            if op.disabled:
                continue
            # pos_info = self.pos_info(op_sched, i)
            op_type = op.__class__.__name__
            if op_type not in self.compile_op:
                raise SyntaxWarning(f"Unrecognized operation type {op_type}")
            op.fct_list = []
            self.compile_op[op_type](op)

    def _activation_placehold(self, prep_op: Op, cluster, output_nodes):
        for anode in cluster.list_anodes:
            if not anode.info or not anode.allocation_type=="data":
                continue  # source anode
            prep_op.add_fct(
                Fct_mem_alloc(
                    anode.main_target,
                    storage=self.storage,
                    shape="Empty_size",
                    dtype=torch.float32,
                    alloc_mode="tensor",
                    requires_grad=anode.info.requires_grad,
                )
            )

        for out_anode in list(cluster.interfaces["output_data_anodes"]):
            for out_target in out_anode.all_targets:
                prep_op.add_fct(
                    Fct_mem_alloc(
                        f"out_{out_target}",
                        storage=self.storage,
                        shape="Empty_size",
                        dtype=torch.float32,
                        alloc_mode="tensor",
                        requires_grad=out_anode.info.requires_grad,
                    )
                )

    def _parameter_placehold(self, prep_op: Op, cluster, minor_param_nodes):
        for pnode in cluster.parameter_nodes:
            ignore = False
            if pnode.is_buffer or pnode in minor_param_nodes:
                device = self.storage.gd["device"]
                ignore = True
            else:
                device = torch.device("cpu")
            prep_op.add_fct(
                Fct_to_storage(
                    pnode.param_name, storage=self.storage, pnode=pnode, device=device
                )
            )

            prep_op.add_fct(
                Fct_run_fwd(
                    pnode.param_name, storage=self.storage, code=pnode.get_code()
                )
            )

            prep_op.add_fct(Fct_get_shape(pnode.param_name, storage=self.storage))
            prep_op.add_fct(Fct_RNG_state(pnode.param_name, storage=self.storage))
            if (
                not ignore
            ):  # TODO: check if cpu preparation is necessary for the parameter
                prep_op.add_fct(
                    Fct_mem_alloc(
                        f"cpu_{pnode.param_name}",
                        storage=self.storage,
                        alloc_mode="tensor",
                        device=torch.device("cpu"),
                        pin_memory=True,
                    )
                )
                prep_op.add_fct(
                    Fct_offload(pnode.param_name, storage=self.storage, pin_memory=True)
                )
                prep_op.add_fct(Fct_del(pnode.param_name, storage=self.storage))
                if pnode.requires_grad:
                    prep_op.add_fct(
                        Fct_mem_alloc(
                            f"cpu_{pnode.param_name}",
                            storage=self.storage,
                            device=torch.device("cpu"),
                            alloc_mode="grad",
                            pin_memory=True,
                        )
                    )

    def _optimizer_placehold(self, prep_op: Op, op_list, minor_param_nodes):
        for op in op_list:
            if isinstance(op, OptimizeOp):
                optim = (
                    self.storage.gd["cpu_optim"]
                    if "cpu" in op.name
                    else self.storage.gd["gpu_optim"]
                )
                prep_op.add_fct(
                    Fct_add_optimizer(
                        op.name,
                        storage=self.storage,
                        list_params=op.list_params,
                        optim=optim,
                        is_cpu=op.is_cpu,
                        **self.storage.gd["opt_kwargs"],
                    )
                )
            if (
                isinstance(op, OffloadOp)
                and isinstance(op.target, Parameter)
                and op.target.is_optim_states
            ):
                var_name = op.target.param_name
                # CPU optim states are placeholder for offload, they are not necessarily attached to the optimizers
                prep_op.add_fct(
                    Fct_mem_alloc(
                        f"exp_avg_{var_name}",
                        storage=self.storage,
                        alloc_mode="tensor",
                        shape=var_name,
                        device=torch.device("cpu"),
                    )
                )
                prep_op.add_fct(
                    Fct_mem_alloc(
                        f"exp_avg_sq_{var_name}",
                        storage=self.storage,
                        alloc_mode="tensor",
                        shape=var_name,
                        device=torch.device("cpu"),
                    )
                )

        if minor_param_nodes:
            minor_parameters = [pnode.param_name for pnode in minor_param_nodes]
            prep_op.add_fct(
                Fct_add_optimizer(
                    "Optimize_minors",
                    storage=self.storage,
                    list_params=minor_parameters,
                    optim=self.storage.gd["gpu_optim"],
                    **self.storage.gd["opt_kwargs"],
                )
            )

    def compile_preparation(self, cluster, op_sched, minor_param_nodes, output_nodes):
        op_list = op_sched.op_list
        init_op_list = op_sched.init_op_list
        prep_op = Op("Preparation")
        self._activation_placehold(prep_op, cluster, output_nodes)
        # self._parameter_placehold(prep_op, cluster, minor_param_nodes)
        self._optimizer_placehold(prep_op, op_list, minor_param_nodes)
        op_sched.init_op_list = init_op_list+ [prep_op]

    def _compute_fwd(self, op: ComputeOp):
        op.add_fct(
            Fct_RNG_state(
                op.name, storage=self.storage, get_state=op.pos_info["first_occurrence"]
            )
        )
        cnode: ComputationNode = op.target
        if not cnode.info.requires_grad:
            op.add_fct(
                Fct_run_fwd(
                    target_name=cnode.main_target,
                    storage=self.storage,
                    code=cnode.get_code(),
                )
            )
            if op.pos_info["first_occurrence"]:
                for target_name in cnode.tensor_targets:
                    op.add_fct(Fct_get_shape(target_name, storage=self.storage))
            return

        inplace_code = make_str_list_assign(
            cnode.inplace_code, force_special_kwargs=not op.pos_info["first_occurrence"]
        )
        body_code = ""
        for bc in cnode.body_code:
            suffix = ""
            if not op.pos_info["first_occurrence"] and (bc[0] in cnode.tensor_targets):
                suffix = ".data"
            body_code += (
                make_str_assign(
                    bc,
                    suffix=suffix,
                    force_special_kwargs=not op.pos_info["first_occurrence"],
                )
                + "\n"
            )
        main_code = make_str_assign(
            cnode.main_code,
            force_special_kwargs=not op.pos_info["first_occurrence"],
        )
        main_code = main_code.replace(cnode.main_target, f"_{cnode.main_target}")
        for pnode in cnode.required_parameter_nodes_real:
            op.add_fct(
                Fct_run_fwd(
                    cnode.main_target,
                    storage=self.storage,
                    code=pnode.get_code(),
                )
            )

        if not op.pos_info["last_before_bwd"]:
            for target in cnode.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)
            op.add_fct(
                Fct_run_fwd(
                    cnode.main_target,
                    storage=self.storage,
                    code=main_code,
                    fwd_mode="no_grad",
                )
            )

        else:
            for target in cnode.tensor_targets:
                inplace_code = inplace_code.replace(target, "_" + target)

            op.add_fct(
                Fct_run_fwd(
                    cnode.main_target,
                    storage=self.storage,
                    code=main_code,
                    no_save_list=op.pos_info["no_save_list"],
                    fwd_mode="with_grad",
                )
            )
        op.add_fct(
            Fct_run_fwd(cnode.main_target, storage=self.storage, code=inplace_code)
        )
        for inplace_target in cnode.inplace_targets:
            if inplace_target != cnode.main_target:
                op.add_fct(
                    Fct_del(f"_{inplace_target}", storage=self.storage, del_mode="data")
                )
        if True:  # TODO:fake detach
            op.add_fct(Fct_detach(cnode.main_target, storage=self.storage))
        op.add_fct(Fct_run_fwd(cnode.main_target, storage=self.storage, code=body_code))
        if op.pos_info["first_occurrence"]:
            for target_name in cnode.tensor_targets:
                op.add_fct(Fct_get_shape(target_name, storage=self.storage))

    def _compute_bwd(self, op: ComputeOp):
        cnode: ComputationNode = op.target
        delete_tensor_function_list = []
        op.add_fct(
            Fct_RNG_state(
                op.name, storage=self.storage, get_state=op.pos_info["first_occurrence"]
            )
        )

        for pnode in cnode.required_parameter_nodes_real:
            op.add_fct(
                Fct_run_fwd(
                    cnode.main_target,
                    storage=self.storage,
                    code=pnode.get_code(),
                )
            )

        for target_name in op.pos_info["temporary_tensor_names"]:
            op.add_fct(
                Fct_gen_fake_data(
                    target_name,
                    storage=self.storage,
                    with_proxy=(cnode.main_target == target_name),
                )
            )
            delete_tensor_function_list.append(
                Fct_del(target_name, storage=self.storage, del_mode="data")
            )
            if cnode.main_target == target_name:
                delete_tensor_function_list.append(
                    Fct_del(f"_{target_name}", storage=self.storage, del_mode="data")
                )

        op.add_fct(
            Fct_run_bwd(
                target_name=cnode.main_target,
                storage=self.storage,
                retain_graph=(not op.pos_info["last_occurrence"]),
                input_names=op.pos_info["input_names"],
            )
        )

        for f in delete_tensor_function_list:
            op.add_fct(f)

    def Compute(self, op: ComputeOp):
        if op.target.is_fwd:
            self._compute_fwd(op)
        else:
            self._compute_bwd(op)

    def Delete(self, op: DeleteOp):
        if isinstance(op.target, Activation):
            alloc: Activation = op.target
            if alloc.anode.allocation_type == "grad":
                op.add_fct(
                    Fct_del(
                        target_name=alloc.anode.main_target,
                        storage=self.storage,
                        del_mode=alloc.anode.allocation_type,
                    )
                )
            elif alloc.anode.allocation_type == "data":
                op.add_fct(
                    Fct_del(
                        target_name=alloc.anode.main_target,
                        storage=self.storage,
                        del_mode=alloc.anode.allocation_type,
                    )
                )
                if alloc.anode.info is not None and alloc.anode.info.requires_grad:
                    op.add_fct(
                        Fct_del(
                            target_name=f"_{alloc.anode.main_target}",
                            storage=self.storage,
                            del_mode="data",
                        )
                    )
                if alloc.anode.has_attribute__base:
                    op.add_fct(
                        Fct_del(
                            target_name=alloc.anode.main_target,
                            storage=self.storage,
                            del_mode="base",
                        )
                    )
                for v in alloc.anode.tensor_targets:
                    op.add_fct(
                        Fct_del(target_name=v, storage=self.storage, del_mode="data")
                    )
                for v in alloc.anode.container_targets:
                    op.add_fct(
                        Fct_del(target_name=v, storage=self.storage, del_mode="var")
                    )
        elif isinstance(op.target, Parameter):
            alloc: Parameter = op.target
            del_mode = "data"
            if alloc.is_grad:
                del_mode = "grad"
            if alloc.is_optim_states:
                del_mode = "optim_states"
            op.add_fct(
                Fct_del(
                    target_name=alloc.param_name,
                    storage=self.storage,
                    del_mode=del_mode,
                )
            )
            if del_mode == "data":
                for target in op.target.pnode.view_targets:
                    op.add_fct(
                    Fct_del(
                        target_name=target,
                        storage=self.storage,
                        del_mode="data",
                    )
                )       

    def Offload(self, op: OffloadOp):
        target: Parameter = op.target
        offload_mode = "param"
        if target.is_grad:
            offload_mode = "grad"
        if target.is_optim_states:
            offload_mode = "optim_states"
        op.add_fct(
            Fct_offload(
                target.param_name, storage=self.storage, offload_mode=offload_mode
            )
        )

    def Prefetch(self, op: PrefetchOp):
        target: Parameter = op.target
        prefetch_mode = "param"
        if target.is_optim_states:
            prefetch_mode = "optim_states"
        op.add_fct(
            Fct_prefetch(target, storage=self.storage, prefetch_mode=prefetch_mode)
        )
        pass

    def Allocate(self, op: AllocateOp):
        target: Parameter = op.target
        alloc_mode = "data"
        if target.is_optim_states:
            alloc_mode = "optim_states"
        op.add_fct(
            Fct_mem_alloc(
                target.param_name, storage=self.storage, alloc_mode=alloc_mode
            )
        )

    def Optimize(self, op: OptimizeOp):
        op.add_fct(
            Fct_optimize(
                op.name,
                storage=self.storage,
                del_grad_list=op.list_params if op.is_cpu else [],
            )
        )

    def Synchronize(self, op: SynchronizeOp):
        op.add_fct(Fct_synchronize(storage=self.storage))

    def Prepare(self, op:PrepareOp):
        pnode = op.target.pnode
        on_cpu = op.device=="cpu"
        op.add_fct(Fct_get_shape(pnode.param_name, pnode=pnode, storage=self.storage))

        if op.cpu_placeholder:
            op.add_fct(
                    Fct_mem_alloc(
                        f"cpu_{pnode.param_name}",
                        storage=self.storage,
                        alloc_mode="tensor",
                        device=torch.device("cpu"),
                        pin_memory=True,
                    )
                )

        op.add_fct(
                Fct_to_storage(
                    "cpu_"*on_cpu + pnode.param_name, 
                    storage=self.storage, 
                    pnode=pnode, 
                    device=op.device
                )
            )
        
        # op.add_fct(Fct_RNG_state(pnode.param_name, storage=self.storage))
        
        if not on_cpu:
            op.add_fct(
                Fct_run_fwd(
                    pnode.param_name, storage=self.storage, code=pnode.get_code()
                )
            )
            if op.cpu_placeholder:
                op.add_fct(
                    Fct_offload(pnode.param_name, storage=self.storage, pin_memory=True)
                )
        else:
            op.add_fct(Fct_del(pnode.param_name, storage=self.storage))

        if pnode.requires_grad:#TODO: check if need cpu_grad placeholder
            op.add_fct(
                Fct_mem_alloc(
                    f"cpu_{pnode.param_name}",
                    storage=self.storage,
                    device=torch.device("cpu"),
                    alloc_mode="grad",
                    pin_memory=True,
                )
            )

    def get_val(self, val):
        if val in self.storage.ld:
            return self.storage.ld[val]
        elif val in self.storage.gd:
            return self.storage.gd[val]
        else:
            raise Exception(f"{val} not defined in executing RK_Env")
