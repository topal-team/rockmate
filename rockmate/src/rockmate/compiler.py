from rkgb.lowlevel.ast_add_on import (
    make_str_assign,
    make_str_list_assign,
    ast_to_str,
    make_ast_list_assign,
)
from rkgb.lowlevel.constants import float_dtype
from rkgb.core.backward import ComputationNode, AllocationNode
import torch
import math
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
        "meta": {dtype: torch.ones(1, dtype=dtype, device=device, requires_grad=True) for dtype in float_dtype},
        # "cpu_optim": optimize_metrics["cpu_optim"],
        # "gpu_optim": optimize_metrics["gpu_optim"],
        "optimize_metrics": optimize_metrics,
        "main_stream": torch.cuda.current_stream(),
        # "prefetch_stream": torch.cuda.current_stream(),
        # "offload_stream": torch.cuda.current_stream(),
        "prefetch_stream": torch.cuda.Stream(device),
        "offload_stream": torch.cuda.Stream(device),
        "cpu_stream": torch.cuda.Stream(device),
    }


class RK_Storage:
    """
    There can only be one RK_Storage.
    All the changes will be made correspond to it.
    All the RK_Fct share the storage.
    """

    _instance = None

    def __init__(self):
        self.batch_multiplier = 1
        self.dynamic_shapes = {}
        self.dict_info = {}
        pass

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def init(self, gd, dtype=None):
        self.ld = dict()
        self.measured_shapes = {"Empty_size": torch.Size([0])}
        self.dtypes = dict()
        self.rng_state = RngState()
        self.events = dict()
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype
        self.gd = gd
        self.manager = AllocationManager(self, dtype=dtype)

    def add_val(self, target, x):
        self.ld[target] = x

    def get_val(self, target):
        if target in self.manager.target_allocations:
            return self.manager.get_val(target)
        if target in self.ld:
            return self.ld[target]
        elif target in self.gd:
            return self.gd[target]
        else:
            raise Exception(f"{target} not defined in executing RK_Env")

    def get_shape(self, target, shape=None):
        if target in self.measured_shapes:
            return self.measured_shapes[target]
        if target in self.dynamic_shapes:
            shape = self.dynamic_shapes[target]
            l = list(shape)
            for i, dim in enumerate(l):
                if isinstance(dim, torch.SymInt):
                    l[i] = int(
                        self.batch_multiplier * self.dict_info[target].tensor_size[i]
                    )
            shape = torch.Size(l)
        elif target in self.dict_info:
            shape = self.dict_info[target].tensor_size
        if shape is None:
            raise ValueError(f"shape of {target} is unkown")
        return shape


class AllocationManager:
    """
    Allocation manager will alloc added tensors together which reduces RAM usage.
    It also keeps the index of tensors which allows to retrieve allocation as a slice
    """

    def __init__(
        self, storage: RK_Storage, dtype=torch.float32, minor_size=10 * 1024**2
    ) -> None:
        self.size = 0
        self.storage = storage
        self.index = 0
        self.allocations = []
        self.target_indices = {}
        self.target_allocations = {}
        self.target_sizes = {}
        self.dtype = dtype
        self.minor_size = minor_size

    def add_tensor(self, target_name, shape):
        self.target_sizes[target_name] = shape.numel()
        self.index += shape.numel()

    def get_val(self, val):
        alloc = self.allocations[self.target_allocations[val]]
        return self.storage.ld[alloc[0]][
            self.target_indices[val][0] : self.target_indices[val][1]
        ]

    def create_allocation(self, size):
        i = len(self.allocations)
        self.allocations.append((f"allocation_{i}", size))

    def add_tensor_to_allocation(self, target, allocation):
        self.target_allocations[target] = allocation

    def split(self):
        targets = list(
            sorted(self.target_sizes.keys(), key=lambda x: self.target_sizes[x])
        )
        remain_size = sum(self.target_sizes[target] for target in targets)
        while remain_size > 0:
            i = len(self.allocations)
            n = int(math.log(remain_size, 2))
            if self.target_sizes[targets[-1]] >= 2**n or remain_size < self.minor_size:
                alloction_size = remain_size
                alloc_targets = targets
            else:
                alloction_size = 0
                alloc_targets = []
                while alloction_size + self.target_sizes[targets[-1]] < 2**n:
                    target = targets.pop()
                    alloction_size += self.target_sizes[target]
                    alloc_targets.append(target)

            self.create_allocation(alloction_size)
            index = 0
            for target in alloc_targets:
                self.target_allocations[target] = i
                self.target_indices[target] = (index, index + self.target_sizes[target])
                index += self.target_sizes[target]
            remain_size -= alloction_size


class RK_Fct:
    def __init__(self, target_name: str, storage: RK_Storage, **kwargs):
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
        self.storage.get_val(self.target_name).data = torch.empty(0, 
                                                                  dtype=self.storage.dtype,
                                                                  device=self.storage.gd["device"])
        # pass

    def del_grad(self):
        self.storage.get_val(self.target_name).grad = None

    def del_base(self):
        if self.storage.get_val(self.target_name)._base is None:
            return
        self.storage.get_val(self.target_name)._base.data = torch.empty(0, 
                                                                        dtype=self.storage.dtype,
                                                                        device=self.storage.gd["device"])

    def del_var(self):
        self.storage.ld[self.target_name] = torch.empty(0, 
                                                        dtype=self.storage.dtype,
                                                        device=self.storage.gd["device"])

    def del_optim_states(self):
        for k, v in self.storage.get_val(
            f"Optimize_({self.target_name})"
        ).state.items():
            v["exp_avg"].data = torch.empty(0, dtype=self.storage.dtype, device=self.storage.gd["device"])
            v["exp_avg_sq"].data = torch.empty(0, dtype=self.storage.dtype, device=self.storage.gd["device"])


class Fct_gen_fake_data(RK_Fct):
    def __init__(
        self, target_name: str, storage: RK_Storage, with_proxy=False, **kwargs
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.with_proxy = with_proxy

    def __call__(self):
        m = self.storage.gd["meta"][self.storage.dtypes[self.target_name]]
        s = self.storage.get_shape(self.target_name)
        if s == torch.Size([]):
            x = m.sum()  # easy way to obtain a Tensor of shape []
        else:
            x = m.expand(*s)
        self.storage.get_val(self.target_name).data = x
        if self.with_proxy:
            self.storage.get_val(f"_{self.target_name}").data = x


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
        self.storage.add_val(
            self.target_name, self.storage.get_val(f"_{self.target_name}")
        )
        self.storage.add_val(f"_{self.target_name}", torch.empty(0, 
                                                                 dtype=self.storage.dtype, 
                                                                 device=self.storage.gd["device"]))

    def __call__(self):
        self.storage.get_val(self.target_name).data = self.storage.ld[
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
        inputs = [self.storage.get_val(name) for name in self.input_names]
        if not inputs:
            inputs = None
        self.storage.get_val(f"_{self.target_name}").backward(
            self.storage.get_val(self.target_name).grad,
            inputs=inputs,
            retain_graph=self.retain_graph,
        )


class Fct_run_fwd(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        code,
        stream="main_stream",
        no_save_list=[],
        fwd_mode="with_grad",
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.str_code = code
        self.code = compile(code, '<string>', "exec")
        self.no_save_list = no_save_list
        self.fwd_fct = {"with_grad": self.fwd_with_grad, "no_grad": self.fwd_no_grad}
        self.fwd_mode = fwd_mode
        self.stream = stream

    def fwd_with_grad(self):
        if self.no_save_list:
            with torch.autograd.graph.saved_tensors_hooks(
                self.fct_get_pack(self.no_save_list), self.fct_get_unpack()
                ):
                exec(self.code, self.storage.gd, self.storage.ld)
        else:
            exec(self.code, self.storage.gd, self.storage.ld)

    def fwd_no_grad(self):
        with torch.no_grad():
            exec(self.code, self.storage.gd, self.storage.ld)

    def __call__(self):
        # with torch.cuda.stream(self.storage.gd[self.stream]):
        self.fwd_fct[self.fwd_mode]()

    def fct_get_pack(self, no_save_list, sanity_check=False):
        no_save_dict = {self.storage.get_val(c).data_ptr() : c
                        for c in no_save_list}
        # no_save_list contains a list of names
        def pack(x):
            c = no_save_dict.get(x.data_ptr(), None)
            if c is not None:
                if sanity_check:
                    assert torch.equal(
                        self.storage.get_val(c).data.as_strided_(
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
                target = self.storage.get_val(x[0])
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
            target = self.storage.get_val(self.target_name)

        self.storage.measured_shapes[self.target_name] = target.shape
        self.storage.measured_shapes[f"cpu_{self.target_name}"] = target.shape
        self.storage.dtypes[self.target_name] = target.dtype
        self.storage.dtypes[f"cpu_{self.target_name}"] = target.dtype


class Fct_optimize(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        del_grad_list: list = [],
        stream="main_stream",
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.del_grad_list = del_grad_list
        self.stream = stream

    def __call__(self):
        # if "cpu" in self.target_name:
        #     return None
        # with torch.cuda.stream(self.storage.gd[self.stream]):
        self.storage.get_val(self.target_name).step()
        # for p in self.del_grad_list:
        #     self.storage.get_val(p).grad.zero_()


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
        shape = self.storage.get_shape(self.shape_name)
        for k, v in self.storage.get_val(
            f"Optimize_({self.target_name})"
        ).state.items():
            v["exp_avg"].data = torch.empty(shape, dtype=self.dtype, device=self.device)
            v["exp_avg_sq"].data = torch.empty(shape, dtype=self.dtype, device=self.device)

    def alloc_grad(self):
        shape = self.storage.get_shape(self.shape_name)
        dtype = (
            self.storage.dtypes[self.shape_name] if self.shape_name in self.storage.dtypes else self.dtype
        )
        self.storage.get_val(self.target_name).grad = torch.empty(
            shape, dtype=dtype, device=self.device, **self.kwargs
        )

    def alloc_data(self):
        shape = self.storage.get_shape(self.shape_name)
        dtype = (
            self.storage.dtypes[self.shape_name] if self.shape_name in self.storage.dtypes else self.dtype
        )
        self.storage.get_val(self.target_name).data = torch.empty(
            shape, dtype=dtype, device=self.device, **self.kwargs
        )

    def alloc_tensor(self):
        shape = self.storage.get_shape(self.shape_name)
        dtype = (
            self.storage.dtypes[self.shape_name] if self.shape_name in self.storage.dtypes else self.dtype
        )
        self.storage.add_val(
            self.target_name,
            torch.empty(shape, dtype=dtype, device=self.device, **self.kwargs),
        )

    def __call__(self):
        self.alloc_fct[self.alloc_mode]()


class Fct_offload(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        offload_mode="tensor",
        stream="offload_stream",
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.offload_fct = {
            "tensor": self.offload_tensor,
            "grad": self.offload_grad,
            "optim_states": self.offload_optim_states,
        }
        self.offload_mode = offload_mode
        self.stream = stream

    def offload_tensor(self):
        self.storage.get_val(f"cpu_{self.target_name}").data.copy_(
            self.storage.get_val(self.target_name).data.reshape(
                self.storage.get_val(f"cpu_{self.target_name}").shape
            ),
            non_blocking=True,
        )
        pass

    def offload_grad(self):
        self.storage.get_val(f"cpu_{self.target_name}").grad.data.copy_(
            self.storage.get_val(self.target_name).grad.view(
                self.storage.get_val(f"cpu_{self.target_name}").shape
            ),
            non_blocking=True,
        )
        pass

    def offload_optim_states(self):
        for k, v in self.storage.get_val(
            f"Optimize_({self.target_name})"
        ).state.items():
            self.storage.get_val(f"exp_avg_{self.target_name}").copy_(
                v["exp_avg"].reshape(
                    self.storage.get_val(f"exp_avg_{self.target_name}").shape
                ),
                non_blocking=True,
            )
            self.storage.get_val(f"exp_avg_sq_{self.target_name}").copy_(
                v["exp_avg_sq"].reshape(
                    self.storage.get_val(f"exp_avg_sq_{self.target_name}").shape
                ),
                non_blocking=True,
            )
        pass

    def __call__(self):
        with torch.cuda.stream(self.storage.gd[self.stream]):
            pass
            self.offload_fct[self.offload_mode]()


class Fct_prefetch(RK_Fct):
    def __init__(
        self,
        target: Allocation,
        storage: RK_Storage,
        prefetch_mode="tensor",
        stream="prefetch_stream",
        with_proxy=False,
        **kwargs,
    ):
        super().__init__(target_name=target.name, storage=storage)
        self.target = target
        self.target_name = target.target_name
        self.prefetch_fct = {
            "tensor": self.prefetch_tensor,
            "optim_states": self.prefetch_optim_states,
        }
        self.post_process = {
            "tensor": self.post_process_tensor,
            "optim_states": self.post_process_optim_states,
        }
        self.with_proxy = with_proxy
        self.prefetch_mode = prefetch_mode
        self.stream = stream
        self.post_process_code = ""
        if isinstance(target, Parameter):
            self.post_process_code = target.pnode.get_code()

    def prefetch_tensor(self):
        self.storage.get_val(self.target_name).data.copy_(
            self.storage.get_val(f"cpu_{self.target_name}").view(
                self.storage.get_val(self.target_name).shape
            ),
            non_blocking=True,
        )
        if self.with_proxy:
            self.storage.get_val(f"_{self.target_name}").data = self.storage.get_val(
                f"{self.target_name}"
            ).data

    def prefetch_optim_states(self):
        for k, v in self.storage.get_val(
            f"Optimize_({self.target_name})"
        ).state.items():
            v["exp_avg"].copy_(
                self.storage.get_val(f"exp_avg_{self.target_name}").reshape(
                    v["exp_avg"].shape
                ),
                non_blocking=True,
            )
            v["exp_avg_sq"].copy_(
                self.storage.get_val(f"exp_avg_sq_{self.target_name}").reshape(
                    v["exp_avg_sq"].shape
                ),
                non_blocking=True,
            )
        pass

    def post_process_tensor(self):
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


class Fct_record_event(RK_Fct):
    def __init__(self, target_name, stream: str, storage: RK_Storage, **kwargs):
        super().__init__(target_name, storage, **kwargs)
        self.stream = stream

    def __call__(self):
        with torch.cuda.stream(self.storage.gd[self.stream]):
            pass
            self.storage.events[self.target_name] = torch.cuda.Event()
            self.storage.events[self.target_name].record(self.storage.gd[self.stream])


class Fct_wait_event(RK_Fct):
    def __init__(self, target_name, stream: str, storage: RK_Storage, **kwargs):
        super().__init__(target_name, storage, **kwargs)
        self.stream = stream

    def __call__(self):
        self.storage.events[self.target_name].wait(self.storage.gd[self.stream])


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
    Get parameter value from nn.module (storage.gd['self']),
    send to the value in storage.get_val(target_name).
    By default, module parameters will be kept in CPU (f'cpu_{target_name}')
    and the real value should be empty.
    """

    def __init__(
        self,
        target_name,
        storage: RK_Storage,
        pnode,
        device=None,
        pin_memory=True,
        create_tensor = True,
        **kwargs,
    ):
        super().__init__(target_name=target_name, storage=storage, **kwargs)
        self.target_name = target_name
        self.pnode = pnode
        self.device = device
        self.pin_memory = pin_memory
        self.create_tensor = create_tensor

    def __call__(self):
        self.storage.add_val(
            self.pnode.param_name, self.pnode.get_value(self.storage.gd["self"])
        )
        w = self.storage.get_val(self.pnode.param_name)
        if self.create_tensor:
            x = torch.empty_like(w, pin_memory=self.pin_memory)
            self.storage.add_val(f"cpu_{self.pnode.param_name}", torch.nn.Parameter(x))


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
        self.storage.add_val(
            self.target_name,
            self.optim(
                [
                    self.storage.get_val(self.is_cpu * "cpu_" + p)
                    for p in self.list_params
                ],
                **self.kwargs,
            ),
        )


class Fct_add_tensor(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        shape=False,
        dynamic_shape=False,
        **kwargs,
    ):
        super().__init__(
            target_name=target_name,
            storage=storage,
        )
        self.target_name = target_name
        self.kwargs = kwargs
        self.shape_name = shape
        self.dynamic_shape = dynamic_shape

    def __call__(self):
        shape = self.storage.get_shape(self.shape_name)
        self.storage.manager.add_tensor(self.target_name, shape)


class Fct_manager_alloc(RK_Fct):
    def __init__(
        self,
        target_name: str,
        storage: RK_Storage,
        **kwargs,
    ):
        super().__init__(
            target_name=target_name,
            storage=storage,
        )
        self.target_name = target_name
        self.kwargs = kwargs

    def __call__(self):
        self.storage.manager.split()
        for alloc, size in self.storage.manager.allocations:
            self.storage.add_val(
                alloc, torch.empty(size, 
                                   device="cpu", 
                                   pin_memory=True, 
                                   dtype=self.storage.manager.dtype)
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

    def __init__(self, storage: RK_Storage):
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
            "ExecCodeOp": self.ExecCode,
            "Op": lambda x: None,  # only for preparation
        }

    def compile_sched(self, op_sched: OpSchedule):
        self.with_parameters = op_sched.with_parameters
        op_sched.add_pos_info()
        for i, op in enumerate(op_sched.op_list):
            if op.disabled:
                continue
            # pos_info = self.pos_info(op_sched, i)
            op_type = op.__class__.__name__
            if op_type not in self.compile_op:
                raise SyntaxWarning(f"Unrecognized operation type {op_type}")
            op.fct_list = []
            stream = "main_stream"
            if isinstance(op, OffloadOp):
                stream = "offload_stream"
            if isinstance(op, PrefetchOp):
                stream = "prefetch_stream"
            for _op_type, target_name in op.wait_events:
                op.add_fct(
                    Fct_wait_event(f"{_op_type}({target_name})", stream, self.storage)
                )
            self.compile_op[op_type](op)
            if op.record_event:
                op.add_fct(
                    Fct_record_event(
                        f"{op.op_type}({op.target.name})", stream, self.storage
                    )
                )

    def _activation_placehold(self, prep_op: Op, cluster, output_nodes):
        for anode in cluster.list_anodes:
            if not anode.info or not anode.allocation_type == "data":
                continue  # source anode
            prep_op.add_fct(
                Fct_mem_alloc(
                    anode.main_target,
                    storage=self.storage,
                    shape="Empty_size",
                    dtype=self.storage.dtype,
                    alloc_mode="tensor",
                    requires_grad=anode.info.requires_grad,
                )
            )
            if OffloadOp(Activation(anode)).name in self.ops:  # offload activation
                dtype = self.storage.dict_info[anode.main_target].dtype
                if dtype == self.storage.dtype:
                    prep_op.add_fct(
                        Fct_add_tensor(
                            f"cpu_{anode.main_target}",
                            self.storage,
                            shape=anode.main_target,
                            dynamic_shape=self.storage.dynamic_shapes,
                        )
                    )
                else:
                    prep_op.add_fct(
                        Fct_mem_alloc(
                            f"cpu_{anode.main_target}",
                            storage=self.storage,
                            shape=anode.main_target,
                            dtype=dtype,
                            alloc_mode="tensor",
                            device=torch.device("cpu"),
                        )
                )

        for out_anode in list(cluster.interfaces["output_data_anodes"]):
            for out_target in out_anode.all_targets:
                prep_op.add_fct(
                    Fct_mem_alloc(
                        f"out_{out_target}",
                        storage=self.storage,
                        shape="Empty_size",
                        dtype=self.storage.dtype,
                        alloc_mode="tensor",
                        requires_grad=out_anode.info.requires_grad,
                    )
                )

    def _optimizer_placehold(self, prep_op: Op, op_list, minor_param_nodes):
        for op in op_list:
            if isinstance(op, OptimizeOp):
                optim = (
                    self.storage.gd["optimize_metrics"]["cpu_optim"]
                    if "cpu" in op.name
                    else self.storage.gd["optimize_metrics"]["gpu_optim"]
                )
                prep_op.add_fct(
                    Fct_add_optimizer(
                        op.name,
                        storage=self.storage,
                        list_params=op.list_params,
                        optim=optim,
                        is_cpu=op.is_cpu,
                        **self.storage.gd["optimize_metrics"]["optim_kwargs"],
                    )
                )
            if (
                isinstance(op, OffloadOp)
                and isinstance(op.target, Parameter)
                and op.target.is_optim_states
            ):
                var_name = op.target.param_name
                # CPU optim states are placeholder for offload,
                # they are not attached to the optimizers
                prep_op.add_fct(
                    Fct_add_tensor(f"exp_avg_{var_name}", self.storage, shape=var_name)
                )
                prep_op.add_fct(
                    Fct_add_tensor(
                        f"exp_avg_sq_{var_name}", self.storage, shape=var_name
                    )
                )

        if minor_param_nodes:
            minor_parameters = [pnode.param_name for pnode in minor_param_nodes]
            prep_op.add_fct(
                Fct_add_optimizer(
                    "Optimize_minors",
                    storage=self.storage,
                    list_params=minor_parameters,
                    optim=self.storage.gd["optimize_metrics"]["gpu_optim"],
                    **self.storage.gd["optimize_metrics"]["optim_kwargs"],
                )
            )

    def compile_preparation(self, cluster, op_sched, minor_param_nodes, output_nodes):
        op_list = op_sched.op_list
        self.ops = {op.name: op for op in op_list}
        self.post_op_list = []
        init_op_list = op_sched.init_op_list
        for i, op in enumerate(op_sched.init_op_list):
            self.compile_op[op.__class__.__name__](op)
        prep_op = Op("Preparation")
        self._activation_placehold(prep_op, cluster, output_nodes)
        if op_sched.with_parameters and self.storage.gd["optimize_metrics"]:
            self._optimizer_placehold(prep_op, op_list, minor_param_nodes)
        prep_op.add_fct(Fct_manager_alloc("allocation", self.storage))
        op_sched.init_op_list = init_op_list + [prep_op] + self.post_op_list

    def _compute_fwd(self, op: ComputeOp):
        op.add_fct(
            Fct_RNG_state(
                op.name, storage=self.storage, get_state=op.pos_info["first_occurrence"]
            )
        )
        cnode: ComputationNode = op.target
        for pnode in cnode.required_parameter_nodes_real:
            view_code = pnode.get_code()
            if view_code:
                op.add_fct(
                    Fct_run_fwd(
                        cnode.main_target,
                        storage=self.storage,
                        code=view_code,
                    )
                )

        for anode in cnode.deps_real:
            for dep_cnode in anode.deps:
                code = dep_cnode.make_body_code_ast()
                ast_view_code = make_ast_list_assign(code)
                if ast_to_str(ast_view_code):
                    op.add_fct(
                        Fct_run_fwd(
                            op.target,
                            storage=self.storage,
                            code=ast_to_str(ast_view_code),
                            stream="main_stream",
                        )
                    )

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
        if inplace_code:
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
        if body_code:
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
            view_code = pnode.get_code()
            if view_code:
                op.add_fct(
                    Fct_run_fwd(
                        cnode.main_target,
                        storage=self.storage,
                        code=view_code,
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
        offload_mode = "tensor"
        if isinstance(op.target, Parameter):
            # target_name = target.param_name
            if target.is_grad:
                offload_mode = "grad"
            if target.is_optim_states:
                offload_mode = "optim_states"
        # else:
        #     target_name = target.main_target
        op.add_fct(
            Fct_offload(
                target.target_name, storage=self.storage, offload_mode=offload_mode
            )
        )

    def Prefetch(self, op: PrefetchOp):
        target: Parameter = op.target
        prefetch_mode = "tensor"
        if isinstance(op.target, Parameter):
            if target.is_optim_states:
                prefetch_mode = "optim_states"
        op.add_fct(
            Fct_prefetch(
                target,
                storage=self.storage,
                prefetch_mode=prefetch_mode,
                with_proxy=isinstance(op.target, Activation)
                and op.target.info.requires_grad,
            )
        )
        if isinstance(op.target, Activation):
            pass
            # for cnode in op.target.anode.deps:
            #     code = cnode.make_body_code_ast()
            #     ast_view_code = make_ast_list_assign(code)
            #     op.add_fct(Fct_run_fwd(op.target,
            #                            storage=self.storage,
            #                            code=ast_to_str(ast_view_code),
            #                            stream="prefetch_stream"))

        if isinstance(op.target, Parameter) and not (
            op.target.is_grad or op.target.is_optim_states
        ):
            pass
            # op.add_fct(Fct_run_fwd(op.target,
            #                            storage=self.storage,
            #                            code=op.target.pnode.get_code(),
            #                            stream="prefetch_stream"))
        pass

    def Allocate(self, op: AllocateOp):
        target: Parameter = op.target
        alloc_mode = "data"
        if isinstance(op.target, Parameter):
            if target.is_optim_states:
                alloc_mode = "optim_states"
        op.add_fct(
            Fct_mem_alloc(
                target.target_name, storage=self.storage, alloc_mode=alloc_mode, dtype=self.storage.dtype
            )
        )

    def Optimize(self, op: OptimizeOp):
        op.add_fct(
            Fct_optimize(
                op.name,
                storage=self.storage,
                del_grad_list=op.list_params if op.is_cpu else [],
                stream="main_stream",  # "cpu_stream" if "cpu" in op.name else "main_stream",
            )
        )

    def Synchronize(self, op: SynchronizeOp):
        if not op.wait_events:
            op.add_fct(Fct_synchronize(storage=self.storage))
        else:
            for e in op.wait_events:
                op.add_fct(
                    Fct_wait_event(
                        f"{e[0]}({e[1]})", stream=op.stream, storage=self.storage
                    )
                )

    def Prepare(self, op: PrepareOp):
        pnode = op.target.pnode
        on_cpu = op.device == "cpu"
        op.add_fct(Fct_get_shape(pnode.param_name, pnode=pnode, storage=self.storage))
        if not op.cpu_optimize:
            op.add_fct(Fct_add_tensor(f"cpu_{pnode.param_name}",
                                      self.storage, 
                                      shape=pnode.param_name))
            op.add_fct(Fct_to_storage(pnode.param_name,
                                           storage=self.storage,
                                           pnode=pnode,
                                           device=op.device,
                                           create_tensor=False))
            post_op = Op(pnode.param_name)
            self.post_op_list.append(post_op)
            prep_op = post_op
        else:
            op.add_fct(
                Fct_to_storage(
                    pnode.param_name,
                    storage=self.storage,
                    pnode=pnode,
                    device=op.device,
                    pin_memory=True,  # op.pin_memory
                )
            )
            op.add_fct(
                Fct_mem_alloc(
                    f"cpu_{pnode.param_name}",
                    storage=self.storage,
                    device=torch.device("cpu"),
                    alloc_mode="grad",
                    pin_memory=True,
                )
            )
            prep_op = op

        prep_op.add_fct(Fct_offload(pnode.param_name, storage=self.storage))
        prep_op.add_fct(Fct_del(pnode.param_name, storage=self.storage))
        if not on_cpu:
            prep_op.add_fct(Fct_mem_alloc(pnode.param_name, storage=self.storage, dtype=self.storage.dtype))
            prep_op.add_fct(Fct_prefetch(op.target, storage=self.storage))
            view_code = pnode.get_code()
            if view_code:
                prep_op.add_fct(
                    Fct_run_fwd(
                        pnode.param_name, storage=self.storage, code=view_code
                    )
                )

    def ExecCode(self, op: ExecCodeOp):
        op.add_fct(Fct_run_fwd(op.name, storage=self.storage, code=op.code))

    def get_val(self, val):
        if val in self.storage.ld:
            return self.storage.ld[val]
        elif val in self.storage.gd:
            return self.storage.gd[val]
        else:
            raise Exception(f"{val} not defined in executing RK_Env")
