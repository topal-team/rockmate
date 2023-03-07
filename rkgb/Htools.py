# ==========================
#  ====== H structure =======
# ==========================

from rkgb.utils import *
from collections import namedtuple

# The H_graph contains only forward part

# ************
# * H_C_node *
# ************


class H_C_node:
    def __init__(self, name):
        """
        Time and overhead information for fast forward.
        In HILP, running HCN means fast forward unless specified otherwise.
        """
        self.name = name  # e.g. Fwd_1
        self.deps = set()  # HDN set
        self.users = set()  # HDN set
        self.sub_graph = None
        self.time = None
        self.overhead = None
        self.is_fwd = True  # if False, self.time=self.overhead=None


# ************
# * H_D_node *
# ************


class H_D_node:
    def __init__(self, name):
        self.name = name
        self.deps = set()  # HCN set
        self.users = set()  # HCN set
        self.mem = 0


# ***********
# * H_option *
# ***********


class H_option:
    """
    An H_option should be useful for several purpose:
        1. it provides the time/memory information for each feasible
        schedule of running forward/backward loop for the corresponding
        H_Graph.
        2. it should be easy to consider the alive status of interface
        HDN's for the accurate overhead
        3. after solving everything, compiler could easily read through
        the H_option and understand what operations should be executed.
    """

    def __init__(self, hgraph, op_list, alive_list):
        self.loss_idx = op_list.index(("Run", "Fwd_loss", 0))
        self.op_list = op_list
        self.alive_list = alive_list
        L = len(op_list)

        def _sum_time(op_list_):
            time = 0
            for (op_type, name, opt) in op_list_:
                if op_type == "Run":
                    hcn = hgraph.dict_hn[name]
                    if opt == len(hcn.sub_graph.list_opt):
                        time += hcn.time
                    else:
                        h_opt = hcn.sub_graph.list_opt[opt]
                        time += h_opt.fwd_time if hcn.is_fwd else h_opt.bwd_time
            return time

        self.fwd_time = _sum_time(op_list[: self.loss_idx])
        self.bwd_time = _sum_time(op_list[self.loss_idx :])
        self.phantom_status = alive_list[self.loss_idx]
        self.save_mem = np.zeros(L)
        self.overhead = np.zeros(L)

        def _sum_mem(alive_status_, ignore_list=[]):
            mem = 0
            for k, v in alive_status_.items():
                if k in ignore_list:
                    continue
                d = hgraph.dict_hn[k]
                if isinstance(d, H_D_node):
                    mem += d.mem if v else 0
                else:
                    mem += d.sub_graph.list_opt[v].save_mem if v > -1 else 0
            return mem

        interfaces_names = []
        for inter in [
            hgraph.fwd_inputs,
            hgraph.bwd_inputs,
            hgraph.fwd_outputs,
            hgraph.bwd_outputs,
        ]:
            interfaces_names += [hdn.name for hdn in inter]

        for i, (op, alive_status) in enumerate(zip(op_list, alive_list)):
            self.save_mem[i] = _sum_mem(alive_status, interfaces_names)
            if op[0] == "Run":
                hcn = hgraph.dict_hn[op[1]]
                if op[2] == len(hcn.sub_graph.list_opt):
                    self.overhead[i] = hcn.overhead
                else:
                    h_opt = hcn.sub_graph.list_opt[op[2]]
                    self.overhead[i] = (
                        h_opt.fwd_overhead if hcn.is_fwd else h_opt.bwd_overhead
                    )
        self.phantom_mem = self.save_mem[self.loss_idx]

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        self.fwd_overhead = get_overhead_(
            self.save_mem[: self.loss_idx + 1],
            self.overhead[: self.loss_idx + 1],
        )
        self.bwd_overhead = get_overhead_(
            self.save_mem[self.loss_idx + 1 :],
            self.overhead[self.loss_idx + 1 :],
        )


H_option = namedtuple(
    "H_option",
    [
        "phantom_mem",
        "fwd_time",
        "bwd_time",
        "fwd_overhead",
        "bwd_overhead",
        "fwd_op_sched",
        "bwd_op_sched",
        "req_inputs",
    ],
)

# ***********
# * H_graph *
# ***********


class H_graph:
    def __init__(self, name):
        """
        All the HCN's and HDN's should be in the same level.
        """
        self.name = name
        self.dict_hn = dict()  #  name -> HN
        self.list_hcn = []  #  toposorted
        self.list_hdn = []  #  including interface HDNs
        self.list_opt = []
        self.fwd_inputs = set()  # HDN set
        self.fwd_outputs = set()  # HDN set
        self.bwd_inputs = set()  # HDN set
        self.bwd_outputs = set()  # HDN set

    def add_option(self, option):
        pareto = True
        for opt in self.list_opt:
            if (
                opt.fwd_time + opt.bwd_time <= option.fwd_time + option.bwd_time
            ) and (opt.save_mem <= option.save_mem):
                # should consider other factors like req_inputs
                pareto = False
        if pareto:
            self.list_opt.append(option)

    def make_users(self):
        for hn in self.dict_hn.values():
            for req_hn in hn.deps:
                req_hn.users.add(hn)
