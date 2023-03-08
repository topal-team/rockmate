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
        self.sub_graph = None  # is None, meaning HCN is fwd and requires no bwd
        self.fwd_time = 0
        self.fwd_overhead = 0
        self.is_fwd = True  # if False, self.fwd_time=self.fwd_overhead=0


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
# * H_graph *
# ***********


class H_graph:
    def __init__(self, name):
        """
        All the HCN's and HDN's should be in the same level.
        When list_hcn and list_hdn are empty, H_graph correspond to
        one pair of fwd/bwd HCN and there is no lower level.
        There should be one H_option and it should be assigned manually.
        """
        self.name = name
        self.dict_hn = dict()  #  name -> HN
        self.dict_hg = dict()  #  name -> sub_Hgraph
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
            ) and (opt.mem <= option.mem):
                # should consider other factors like req_inputs
                pareto = False
        if pareto:
            self.list_opt.append(option)

    def make_users(self):
        for hn in self.dict_hn.values():
            for req_hn in hn.deps:
                req_hn.users.add(hn)


# ***********
# * H_op *
# ***********


class H_op:
    def __init__(self, name, h_obj, is_fwd=True, is_del=False):
        self.name = name
        self.is_fwd = is_fwd
        self.is_del = is_del
        self.obj = h_obj


# ***********
# * H_option *
# ***********


class H_option:
    """
    .mem: phantom memory saved from fwd to bwd
    .fwd_time/bwd_time
    .fwd_overhead/bwd_overhead
    An H_option should be useful for several purpose:
        1. it provides the time/memory information for each feasible
        schedule of running forward/backward loop for the corresponding
        H_Graph.
        2. it should be easy to consider the alive status of interface
        HDN's for the accurate overhead.
        3. after solving everything, compiler could easily read through
        the H_option and understand what operations should be executed.
    """

    def __init__(self, hgraph, op_list, alive_list):
        # when op_list and alive_list are empty, all the information can be
        # assigned directly
        for i, op in enumerate(op_list):
            if "loss" in op.name:
                self.loss_idx = i
                break
        self.op_list = op_list
        self.alive_list = alive_list
        L = len(op_list)
        self.time = np.zeros(L)
        self.save_mem = np.zeros(L)
        self.overhead = np.zeros(L)

        def _sum_mem(alive_status_, ignore_list=[]):
            mem = 0
            for k, v in alive_status_.items():
                if k in ignore_list:
                    continue
                if k in hgraph.dict_hn:
                    d = hgraph.dict_hn[k]
                    mem += d.mem if v > -1 else 0
                else:
                    mem += hgraph.dict_hg[k].list_opt[v].mem if v > -1 else 0
            return mem

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

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
            if not op.is_del:
                if op.is_fwd:
                    self.time[i] = op.fwd_time
                    self.overhead[i] = op.fwd_overhead
                else:
                    self.time[i] = op.bwd_time
                    self.overhead[i] = op.bwd_overhead

        self.mem = self.save_mem[self.loss_idx]
        self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
        self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])
        self.fwd_overhead = get_overhead_(
            self.save_mem[: self.loss_idx + 1],
            self.overhead[: self.loss_idx + 1],
        )
        self.bwd_overhead = get_overhead_(
            self.save_mem[self.loss_idx + 1 :],
            self.overhead[self.loss_idx + 1 :],
        )

        self.dep_inputs = []  # the names of HDNs that are required by BWD
        for op in op_list[self.loss_idx + 1 :]:
            if op.is_run:
                for hdn in hgraph.dict_hn[op.name].deps:
                    if hdn in hgraph.fwd_inputs and hdn.name in self.dep_inputs:
                        self.dep_inputs.append(hdn.name)
