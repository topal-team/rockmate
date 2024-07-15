from psutil import virtual_memory
from typing import Any, Dict
from pulp import lpSum
import math
from rkgb.core.hierarchical import HierarchicalGraph
from .ilp_utils import RkLpVariable
from .ilp_model import ModelPULP


class ModelPULPOffload(ModelPULP):
    def __init__(
        self,
        hgraph: HierarchicalGraph,
        peak_budget: int,
        save_budget=None,
        ilp_solver_params: Dict[str, Any] = ...,
        gcd=None,
        accurate_mem=False,
        protected_names=...,
        grouping=True,
        grad_mode="free", #['free', 'accumulate', 'zero_grad']
        optimize_metrics=None,
        activation_offload=False,
        dynamic_batch_size=False,
        minor_offload_size=None,
        bandwidth=None,
        ram_budget=None,
        cpu_optimize=True,
        **kwargs
    ):
        super().__init__(
            hgraph,
            peak_budget,
            save_budget,
            ilp_solver_params,
            gcd,
            accurate_mem,
            protected_names,
        )

        self.with_grad = accurate_mem
        self.single_fwd = accurate_mem
        self.single_bwd = accurate_mem
        self.grouping = grouping
        self.grad_mode = grad_mode
        self.activation_offload = activation_offload
        self.optimize_metrics = optimize_metrics
        self.use_correction_term = True
        self.cpu_constant_cost = 100
        self.dynamic_batch_size = dynamic_batch_size
        self.ram_budget = ram_budget or virtual_memory().available * 0.9/self.gcd
        self.cpu_optimize = cpu_optimize
        self.minor_offload_size = minor_offload_size
        self.bandwidth = bandwidth
        if not self.optimize_metrics:
            raise UserWarning("Trying to solve ILP with offload but "\
                              "optimization method is not provided."\
                              "This is not implemented yet.")

        if self.grad_mode != "free":
            raise UserWarning("Parameter gradients are not freed during backward, "\
                              "which is not implemented in ILP.")

    def build(self):
        # OVERWRITTING METHOD
        self.config_offload()
        super().add_variables()
        self.add_offload_param_variables()
        if self.activation_offload:
            self.add_offload_activation_variables()
            self.add_offload_activation_constraints()
        super().add_constraints()
        self.add_offload_param_constraints()
        self.add_ram_constraints()
        # super().add_objective()
        self.add_offload_objective()
        self.add_offload_time_constraints()

    def config_offload(self):
        if self.optimize_metrics:
            optimize_metrics = self.optimize_metrics
            self.optimizer_states_factor = optimize_metrics[
                "optimizer_states_factor"
            ]  # *weight size
            self.cpu_optimize_speed = (
                optimize_metrics["cpu_optimize_speed"] / self.gcd
            )  # B/ms
            self.gpu_optimize_speed = (
                optimize_metrics["gpu_optimize_speed"] / self.gcd
            )  # B/ms
            self.optimizer_overhead_factor = optimize_metrics[
                "optimizer_overhead"
            ]  # *weight size
        else:
            self.optimizer_overhead_factor = 0
            self.optimizer_states_factor = 0
            self.cpu_optimize_speed = 0
            self.gpu_optimize_speed = 0

        # self.bandwidth = optimize_metrics["bandwidth"]  # bandwidth
        self.bandwidthOfl = self.bandwidth / self.gcd  # byte/ms
        self.bandwidthPrf = self.bandwidth / self.gcd  # byte/ms
        
        self.schedule_offload_time = []
        self.schedule_prefetch_time = []
        self.schedule_fwd_wait_time = []
        self.schedule_bwd_wait_time = []
        for list_sched in self.list_list_sched:
            self.schedule_offload_time.append([])
            self.schedule_prefetch_time.append([])
            self.schedule_fwd_wait_time.append([])
            self.schedule_bwd_wait_time.append([])
            for sched in list_sched:
                self.schedule_offload_time[-1].append(sched.offload_mem/self.bandwidthOfl / self.gcd)
                self.schedule_prefetch_time[-1].append(sched.prefetch_mem/self.bandwidthPrf / self.gcd)
                self.schedule_fwd_wait_time[-1].append(sched.fwd_wait_time)
                self.schedule_bwd_wait_time[-1].append(sched.bwd_wait_time)

        self.param2hcn = dict()
        self.parameters = []
        for k, v in self.hgraph.parameter_groups.items():
            self.param2hcn[len(self.parameters)] = k
            self.parameters.append(v)

        self.hcn2param = {t: [] for t in range(self.T)}

        for p, hcn_s in self.param2hcn.items():
            for hcn in hcn_s:
                self.hcn2param[hcn].append(p)

        self.parameter_size = [
            sum(pnode.mem for pnode in p) / self.gcd for p in self.parameters
        ]
        self.parameter_gradient_size = [
            sum(pnode.mem for pnode in p if pnode.info.requires_grad) / self.gcd
            for p in self.parameters
        ]
        self.W = len(self.parameters)

    def add_offload_time_constraints(self):
        for t in range(self.T):
            for k in self.krange(t):
                self.md += self.Time[t, k] >= self.time_step_prefetch(t, k)
                self.md += self.Time[t, k] >= (
                    self.time_step_offload(t, k) + self.time_step_optimize_self(t, k)
                )
                self.md += self.Time[t, k] >= self.time_step_optimize(t, k)
                self.md += self.Time[t, k] >= (
                    self.time_step_compute(t, k)
                    + self.time_step_optimize_self(t, k, cpu=False)
                    + self.time_step_offload_self(t, k)
                    + self.time_step_optimize_self(t, k, cpu=True)
                )

                self.md += self.Time[t, k] >= (
                    self.time_step_compute(t, k)+
                    self.time_step_prefetch_self(t,k))

    def add_offload_activation_variables(self):
        """
        We assume single forwarard/backward mode for ILP, thus only
        one possible source/user for phantom activations.
        """
        phantom_idx = [
            (t, k, j)
            for t in range(self.T)
            for k in self.krange(t)
            for j in range(self.J)
        ]
        # at step (t,k), removal of phantom of the j-th node.
        # self.RemovalP = RkLpVariable.dicts("RemovalP", phantom_idx)
        self.OflP = RkLpVariable.dicts("OflP", phantom_idx, lowBound=0, upBound=None)
        self.PrfP = RkLpVariable.dicts("PrfP", phantom_idx, lowBound=0, upBound=None)
        # pass
        self.phantoms = {}
        for j in range(self.J):
            for o, sched in enumerate(self.list_list_sched[j]):
                sub_cluster = self.sub_clusters[j]
                phantoms = [
                    sub_cluster.translate_representee_node(re_anode)
                    for re_anode in sched.phantoms
                    if (
                        re_anode.allocation_type == "data"  # Only support data offload
                        and re_anode.mem > self.minor_offload_size
                    )
                ]
                self.phantoms[(j, o)] = phantoms

    def add_offload_param_variables(
        self,
    ):
        lb = 0 if self.dynamic_batch_size else 1
        self.req_w = RkLpVariable("Required_w", lowBound=lb, upBound=1, cat="Continuous")


        param_idx = [
            (t, k, w)
            for t in range(self.T)
            for k in self.krange(t)
            for w in range(self.W)
        ]

        self.AliveW = RkLpVariable.dicts(
            "AliveW", param_idx
        )  # parameter w is alive at the start of step j.
        self.AliveG = RkLpVariable.dicts(
            "AliveG", param_idx
        )  # w.grad is alive at the start of step j.
        self.AliveO = RkLpVariable.dicts(
            "AliveO", param_idx
        )  # w.grad is alive at the start of step j.
        self.OflW = RkLpVariable.dicts("OflW", param_idx)  # Offload weight
        self.OflG = RkLpVariable.dicts("OflG", param_idx)  # Offload gradient
        self.PrfW = RkLpVariable.dicts("PrfW", param_idx)  # Prefetch gradient
        self.PrfG = RkLpVariable.dicts("PrfG", param_idx)  # Prefetch gradient
        self.OflO = RkLpVariable.dicts("OflO", param_idx, 
                        upBound=1 if self.optimize_metrics else 0)  # Offload optimizer states
        self.PrfO = RkLpVariable.dicts("PrfO", param_idx, 
                        upBound=1 if self.optimize_metrics else 0)  # Prefetch optimizer states

        if self.cpu_optimize and self.optimize_metrics:
            self.OptC = RkLpVariable.dicts("OptC", param_idx)  # Optimize on cpu
        else:
            self.OptC = RkLpVariable.dicts("OptC", param_idx, lowBound=0, upBound=0)
        self.sumOptC = dict()
        for w in self.param2hcn:
            self.sumOptC[w] = lpSum(
                self.OptC[t, k, w] for t in range(self.T) for k in self.krange(t)
            )

        self.param_grad_mem = {(t, k): 0 for t in range(self.T) for k in self.krange(t)}

        self.prefill_offload()

    def accumC_grad(self, w):
        # if grad_accumulation, gradient stored on CPU from previous iterations
        if self.grad_mode == "free":
            return self.sumOptC[w]


    def accumC_optimizer_states(self, w):
        # if grad_accumulation, optimizer states stored on CPU from previous iterations
        return self.accumC_grad(w)

    def max_OflGProg(self, t, k, w):
        return self.OflGProg[t, k, w] + (
            self.OflWProg[t, k, w] * (self.grad_mode == "free") * self.w_by_wg(w)
        )

    def w_by_wg(self, w):
        if self.parameter_gradient_size[w] == 0:
            return self.gcd
        return self.parameter_size[w] / self.parameter_gradient_size[w]

    def activation_mem(self, t, k):
        return self.U[t, k] - self.removal_phantom_mem(t, k)

    def save_mem(self, t, k):
        # OVERWRITTING METHOD
        return self.activation_mem(t, k) + self.all_param_mem(t, k)

    def overhead_mem(self, t, k):
        return self.act_overhead(t, k) + self.param_overhead(t, k)

    def param_overhead(self, t, k):
        optimizer_overhead = 0
        if k > self.loss_idx:  # and k in self.hcn2param:
            l_w = self.hcn2param[k]
            optimizer_overhead += sum(
                (self.req_w - self.sumOptC[w])
                * self.parameter_gradient_size[w]
                * self.optimizer_overhead_factor
                for w in l_w
            )
        return [optimizer_overhead]

    def all_param_mem(self, t, k, with_multiplier=True):
        return (
            self.parameter_mem(t, k)
            + self.param_grad_mem[t, k]
            + self.optimizer_states_mem(t, k)
            + (1 - self.req_w) * self.peak_budget * with_multiplier
        )

    def removal_phantom_mem(self, t, k):
        if not self.activation_offload:
            return 0
        mem = 0
        for j in range(self.J):
            mem += self.OflPProg[t, k, j] - self.PrfPProg[t, k, j]
        return mem

    def parameter_mem(self, t, k):
        parameter_mem = lpSum(
            (self.AliveW[t, k, w] + self.PrfW[t, k, w]) * self.parameter_size[w]
            for w in range(self.W)
        )
        return parameter_mem

    # def param_grad_mem(self,t, k):
    #     grad_mem = lpSum(
    #         self.AliveG[t, k, w]
    #         * self.parameter_gradient_size[w]
    #         for w in range(self.W)
    #     )
    #     return grad_mem

    def optimizer_states_mem(self, t, k, with_overhead=True):
        optimizer_states_mem = lpSum(
            (
                (self.AliveO[t, k, w] + self.PrfO[t, k, w])
                * self.parameter_gradient_size[w]
                * self.optimizer_states_factor
            )
            for w in range(self.W)
        )
        return optimizer_states_mem

    def prefill_offload(
        self,
    ):
        
        for t in range(self.T):
            for k in range(self.T):
                for w in range(self.W):
                    if k not in self.krange(t):
                        self.PrfW[t, k, w] = 0
                        self.OflW[t, k, w] = 0

                    # fwd_i, bwd_i = self.param2hcn[w]
                    fwd_i = min(self.param2hcn[w])
                    bwd_i = max(self.param2hcn[w])
                    if k not in self.krange(t) or (t > fwd_i and t < bwd_i):
                        self.OptC[t, k, w] = 0

        if not self.grad_mode == "accumulate":
            for k in self.PrfG:
                self.PrfG[k] = 0

        if not self.optimizer_states_factor:
            for t, k, w in self.AliveO:
                self.AliveO[(t, k, w)] = self.req_w - self.accumC_optimizer_states(w)
                self.PrfO[(t, k, w)] = 0
                self.OflO[(t, k, w)] = 0
        if self.grad_mode in ["free"]:
            # If gradient is freed ASAP, OflG will be merged into OflW
            for t, k, w in self.OflG:
                self.OflG[(t, k, w)] = 0
            for t, k, w in self.AliveG:
                grad_size = self.parameter_gradient_size[w]
                if grad_size == 0:
                    self.AliveG[(t, k, w)] = 0
                    continue
                if len(self.param2hcn[w]) <= 2:
                    self.AliveG[(t, k, w)] = 0
                    if k == max(self.param2hcn[w]):
                        # self.overhead[k] = [v+grad_size for v in self.overhead[k]]
                        self.param_grad_mem[t, k] += grad_size * self.req_w
                else:  # shared weight
                    bwd_first = min(x for x in self.param2hcn[w] if x > self.loss_idx)
                    bwd_last = max(self.param2hcn[w])
                    if t <= bwd_first or t > bwd_last:  # assume single bwd
                        self.AliveG[(t, k, w)] = 0
                    else:
                        self.AliveG[(t, k, w)] = 1
                        if k in self.param2hcn[w] and k > bwd_first:
                            # self.overhead[k] = [v+grad_size for v in self.overhead[k]]
                            self.param_grad_mem[t, k] += grad_size * self.req_w
        elif self.grad_mode in ["accumulate"]:
            for t, k, w in self.AliveG:
                bwd_indices = [x for x in self.param2hcn[w] if x > self.loss_idx]
                if t in bwd_indices:
                    self.AliveG[(t, k, w)] = 1
                # if k == max(self.param2hcn[w]):
                #     self.overhead[k] = [v+grad_size for v in self.overhead[k]]
                # TODO: add offload gradient variables for gradient accumulation

        for t in range(self.T):
            for k in self.krange(t):
                self.param_grad_mem[t, k] += lpSum(
                    self.AliveG[t, k, w] * self.parameter_gradient_size[w]
                    for w in range(self.W)
                )

    def add_offload_param_constraints(
        self,
    ):
        # if with_grad, AliveG is a variable
        self.OflWProg = dict()
        self.OflGProg = dict()
        self.OptCProg = dict()
        self.PrfWProg = dict()
        self.PrfGProg = dict()
        self.OflOProg = dict()
        self.PrfOProg = dict()

        def get_progress(op, t, k, w):
            bwd_i = max(self.param2hcn[w])
            if bwd_i < t:  # after bwd of w
                progress = (
                    lpSum(op[t, kk, w] for kk in self.krange(t) if kk < k)
                    + lpSum(
                        op[tt, kk, w]
                        for tt in range(bwd_i + 1, t)
                        for kk in self.krange(tt)
                    )
                    + lpSum(
                        op[bwd_i, kk, w] for kk in self.krange(bwd_i) if kk >= bwd_i
                    )  # offload right after bwd_i
                )
            else:
                progress = (
                    lpSum(op[t, kk, w] for kk in self.krange(t) if kk < k)
                    + lpSum(op[tt, kk, w] for tt in range(t) for kk in self.krange(tt))
                    + lpSum(
                        op[tt, kk, w]
                        for tt in range(bwd_i + 1, self.T)
                        for kk in self.krange(tt)
                    )
                    + lpSum(
                        op[bwd_i, kk, w] for kk in self.krange(bwd_i) if kk >= bwd_i
                    )
                )
            return progress

        for t in range(self.T):
            for k in self.krange(t):
                t_, k_ = self.next_idx(t, k)
                # self.md += self.Time[t, k] >= self.time_step_prefetch(t, k)
                # self.md += self.Time[t, k] >= (self.time_step_offload(t, k)
                #                            + self.time_step_optimize_self(t, k))
                # self.md += self.Time[t, k] >= self.time_step_optimize(t, k)
                # self.md += self.Time[t, k] >= (self.time_step_compute(t, k)
                #                            + self.time_step_optimize_self(t, k, cpu=False)
                #                            + self.time_step_offload_self(t, k)
                #                            + self.time_step_optimize_self(t, k, cpu=True))

                for w in range(self.W):
                    self.PrfWProg[t, k, w] = get_progress(self.PrfW, t, k, w)
                    self.PrfGProg[t, k, w] = get_progress(self.PrfG, t, k, w)
                    self.OflWProg[t, k, w] = get_progress(self.OflW, t, k, w)
                    self.OflGProg[t, k, w] = get_progress(self.OflG, t, k, w)
                    self.OflOProg[t, k, w] = get_progress(self.OflO, t, k, w)
                    self.PrfOProg[t, k, w] = get_progress(self.PrfO, t, k, w)
                    self.OptCProg[t, k, w] = get_progress(self.OptC, t, k, w)

                    self.md += self.AliveW[t, k, w] <= self.req_w
                    self.md += self.OflWProg[t, k, w] <= self.req_w
                    self.md += self.OflGProg[t, k, w] <= self.accumC_grad(w)
                    self.md += self.OptCProg[t, k, w] <= self.max_OflGProg(t, k, w)
                    self.md += self.OptCProg[t, k, w] <= self.PrfWProg[
                        t, k, w
                    ] * self.w_by_wg(w)

                    self.md += self.AliveW[t, k, w] + self.OflWProg[
                        t, k, w
                    ] >= self.fraction_constant_param(
                        w
                    ) + self.fraction_instant_updated_param(
                        w
                    ) * (1/self.w_by_wg(w))
                    self.md += self.AliveG[t, k, w] + self.OflGProg[
                        t, k, w
                    ] >= self.fraction_remaining_gradients(w)

                    to_be_prefetched = self.parameter_size[w] * (
                        self.req_w - self.AliveW[t_, k_, w] - self.PrfW[t_, k_, w]
                    )
                    to_be_optimized = self.parameter_gradient_size[w] * (
                        self.sumOptC[w] - self.OptCProg[t, k, w]
                    )
                    self.md += to_be_prefetched >= to_be_optimized

                    diffW = self.AliveW[t_, k_, w] - self.AliveW[t, k, w]
                    self.md += diffW <= self.PrfW[t, k, w]

                    diffG = self.AliveG[t_, k_, w] - self.AliveG[t, k, w]
                    self.md += (
                        diffG
                        <= 1
                        * (
                            k in self.param2hcn[w] and k > self.loss_idx
                        )  # backward operations
                        + self.PrfG[t, k, w]
                    )

                    diffO = self.AliveO[t_, k_, w] - self.AliveO[t, k, w]
                    self.md += diffO <= self.PrfO[t, k, w]

                    self.md += self.OflOProg[t, k, w] >= self.PrfOProg[t, k, w]
                    self.md += self.AliveO[t, k, w] + self.OflOProg[
                        t, k, w
                    ] >= self.fraction_gpu_optimized(w)
                    self.md += self.OflOProg[t, k, w] <= self.fraction_gpu_optimized(w)

        # parameters and their user operations
        for w in self.param2hcn:
            fwd_i = min(self.param2hcn[w])
            bwd_i = max(self.param2hcn[w])
            self.md += self.PrfWProg[fwd_i, fwd_i, w] >= self.sumOptC[w]
            self.md += self.req_w - self.sumOptC[w] <= self.AliveO[bwd_i, bwd_i, w]
            if self.grad_mode == "accumulate":
                self.md += self.OflGProg[bwd_i, bwd_i, w] == self.accumC_grad(w)
                self.md += (
                    self.PrfGProg[bwd_i, bwd_i, w]
                    == self.accumC_grad(w) - self.sumOptC[w]
                )
            t_, k_ = self.next_idx(bwd_i, bwd_i)
            # self.md += self.AliveO[t_, k_, w] - self.AliveO[bwd_i, bwd_i, w] <= 1 - self.grad_mode == "accumulate"
            # self.md += self.AliveG[t_, k_, w] - self.AliveG[bwd_i, bwd_i, w] <= 1# - self.grad_mode == "accumulate"
            for t in range(self.T):
                for k in self.param2hcn[w]:
                    if k not in self.krange(t):
                        continue
                    self.md += (
                        self.sumComp[t, k] - 1 <= self.AliveW[t, k, w] - self.req_w
                    )

    def add_offload_activation_constraints(self):
        self.PrfPProg = dict()
        self.OflPProg = dict()

        def get_progress_phantom(op, t, k, j):
            fwd_i = min(self.sub_c2hcn[j])
            bwd_i = max(self.sub_c2hcn[j])

            if bwd_i < t or fwd_i > t:  # after bwd of w
                return 0

            progress = (
                lpSum(op[t, kk, j] for kk in self.krange(t) if kk < k)
                + lpSum(
                    op[tt, kk, j]
                    for tt in range(fwd_i + 1, t)
                    for kk in self.krange(tt)
                )
                + lpSum(op[fwd_i, kk, j] for kk in self.krange(fwd_i) if kk >= fwd_i)
            )

            return progress

        for j in range(self.J):
            fwd_i = min(self.sub_c2hcn[j])
            bwd_i = max(self.sub_c2hcn[j])
            saved_mem = lpSum(
                # self.saved_mem[j][o]*
                sum(anode.mem for anode in self.phantoms[j, o])
                / self.gcd
                * self.Comp[fwd_i, fwd_i, o]
                for o, sched in enumerate(self.list_list_sched[j])
            )
            for t in range(self.T):
                for k in self.krange(t):
                    if t >= fwd_i and t <= bwd_i:
                        self.PrfPProg[t, k, j] = get_progress_phantom(
                            self.PrfP, t, k, j
                        )
                        self.OflPProg[t, k, j] = get_progress_phantom(
                            self.OflP, t, k, j
                        )

                        # removal cannot be higher than phantom created during fwd
                        self.md += self.OflPProg[t, k, j] <= saved_mem
                    else:
                        self.PrfPProg[t, k, j] = 0
                        self.OflPProg[t, k, j] = 0

            self.md += self.PrfPProg[bwd_i, bwd_i, j] == self.OflPProg[bwd_i, bwd_i, j]

    def add_ram_constraints(self):
        self.md += self.ram_usage() <= self.ram_budget

    def ram_usage(self):
        mem_usage = 0
        # activation on cpu
        for t in range(self.loss_idx):
            for k in self.krange(t):
                j = self.hcn2sub_c[k]
                mem_usage += lpSum(
                    self.Comp[t, k, o] * self.list_list_sched[k][o].offload_mem/self.gcd
                    for o in range(self.nSched[k])
                )
        for w in range(self.W):
            bwd_i = max(self.param2hcn[w])
            mem_usage += self.parameter_size[w] * self.OflWProg[bwd_i, bwd_i, w]
            mem_usage += (self.parameter_gradient_size[w] 
                          * self.OflOProg[bwd_i, bwd_i, w]
                          * self.optimizer_states_factor)
            mem_usage += (self.parameter_gradient_size[w] 
                          * self.sumOptC[w] 
                          * (self.optimizer_states_factor+1))
            
        #parameters are already in RAM
        mem_usage -= sum(self.parameter_size)
        return mem_usage
        

    def add_offload_objective(self, bandwidth_cost=0.01):
        prf_cost = bandwidth_cost * lpSum(
            (self.PrfW[t, k, w] + self.PrfG[t, k, w] + self.PrfO[t, k, w])
            * self.parameter_size[w]
            / self.bandwidthPrf
            for t in range(self.T)
            for k in self.krange(t)
            for w in range(self.W)
        )
        ofl_cost = bandwidth_cost * lpSum(
            (self.OflW[t, k, w] + self.OflG[t, k, w] + self.OflO[t, k, w])
            * self.parameter_size[w]
            / self.bandwidthOfl
            for t in range(self.T)
            for k in self.krange(t)
            for w in range(self.W)
        )
        self.md += (
            lpSum(self.Time[t, k] for t in range(self.T) for k in self.krange(t))
            + prf_cost
            + ofl_cost
        )

    """
    The following functions are meant to be used for building constraints with
    meaningful names. They are not supposed to be called outside this file.
    By default, index t is for stage, k for step and w for the parameter group.
    """

    def fraction_gpu_optimized(self, w):
        return self.req_w - self.sumOptC[w]

    def fraction_constant_param(self, w):
        return 1 - self.parameter_gradient_size[w] / self.parameter_size[w]

    def fraction_instant_updated_param(self, w):
        if self.grad_mode == "accumulate":
            # TODO: not fully supported now
            return 0
        if self.grad_mode == "free":
            return self.req_w
        return self.req_w - self.sumOptC[w]

    def fraction_remaining_gradients(self, w):
        if self.grad_mode == "accumulate":
            # TODO: not fully supported now
            return self.req_w
        if self.grad_mode == "free":
            return 0

    # def fraction_after_bwd(self,w):
    #     return self.fraction_constant_param(w) + self.fraction_instant_updated_param(w)

    def time_step_offload(self, t, k):
        j = self.hcn2sub_c[k]
        sub_sched_time = lpSum(
            self.Comp[t, k, o] * self.schedule_offload_time[j][o]
            for o in range(self.nSched[k])
        )
        mem = 0
        for w in range(self.W):
            mem += self.parameter_size[w] * self.OflW[t, k, w]
            optim_state = self.optimizer_states_factor * self.OflO[t, k, w]
            mem += self.parameter_gradient_size[w] * (self.OflG[t, k, w] + optim_state)
        if self.activation_offload:
            for j in range(self.J):
                mem += self.OflP[t, k, j]

        return mem / self.bandwidthOfl + sub_sched_time

    def time_step_offload_self(self, t, k):
        j = self.hcn2sub_c[k]
        sub_sched_time = lpSum(
            self.Comp[t, k, o] * self.schedule_fwd_wait_time[j][o]
            for o in range(self.nSched[k])
        )
        mem = 0
        for w in self.hcn2param[k]:
            mem += self.parameter_size[w] * self.OflW[t, k, w]
            optim_state = self.optimizer_states_factor * self.OflO[t, k, w]
            mem += self.parameter_gradient_size[w] * (self.OflG[t, k, w] + optim_state)
        if self.activation_offload:
            j = self.hcn2sub_c[k]
            if j is not None:
                mem += self.OflP[t, k, j]

        return mem / self.bandwidthOfl + sub_sched_time

    def time_step_prefetch(self, t, k):
        j = self.hcn2sub_c[k]
        sub_sched_time = lpSum(
            self.Comp[t, k, o] * self.schedule_prefetch_time[j][o]
            for o in range(self.nSched[k])
        )

        mem = 0
        for w in range(self.W):
            mem += self.parameter_size[w] * self.PrfW[t, k, w]
            optim_state = self.optimizer_states_factor * self.PrfO[t, k, w]
            mem += self.parameter_gradient_size[w] * (self.PrfG[t, k, w] + optim_state)
        if self.activation_offload:
            for j in range(self.J):
                mem += self.PrfP[t, k, j]

        return mem / self.bandwidthPrf + sub_sched_time

    def time_step_prefetch_self(self, t, k):
        j = self.hcn2sub_c[k]
        sub_sched_time = lpSum(
            self.Comp[t, k, o] * self.schedule_bwd_wait_time[j][o]
            for o in range(self.nSched[k])
        )
        mem = 0
        for w in self.hcn2param[k]:
            mem += self.parameter_size[w] * self.PrfW[t, k, w]
            optim_state = self.optimizer_states_factor * self.PrfO[t, k, w]
            mem += self.parameter_gradient_size[w] * (self.PrfG[t, k, w] + optim_state)

        return mem / self.bandwidthPrf + sub_sched_time

    def time_step_optimize(self, t, k, cpu=True):
        """
        We assume that GPU optimization happens only right after
        the backward, which should be accessed from time_step_optimize_self.
        """
        if not self.optimize_metrics:
            return 0
        mem = 0
        for w in range(self.W):
            mem += self.parameter_gradient_size[w] * self.OptC[t, k, w]
        return mem / self.cpu_optimize_speed + self.cpu_constant_cost * self.req_w

    def time_step_optimize_self(self, t, k, cpu=True):
        """
        We assume that GPU optimization happens only right after
        the backward, but CPU optimization could happen anytime and
        represented by self.OptC.
        """
        if not self.optimize_metrics:
            return 0
        mem = 0
        if cpu:
            for w in self.hcn2param[k]:
                opt_fraction = self.OptC[t, k, w]
                mem += self.parameter_gradient_size[w] * opt_fraction
            return mem / self.cpu_optimize_speed

        elif k <= self.loss_idx:  # forward
            return 0
        else:
            for w in self.hcn2param[k]:
                opt_fraction = self.fraction_instant_updated_param(w)
                mem += self.parameter_gradient_size[w] * opt_fraction
            return mem / self.gpu_optimize_speed

    def time_step_compute(self, t, k):
        return lpSum(self.Comp[t, k, o] * self.time[k][o] for o in range(self.nComp[k]))
