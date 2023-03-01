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
        self.name = name  # e.g. Fwd_1
        self.deps = set()  # HDN set
        self.users = set()  # HDN set
        self.sub_graph = None


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

H_option = namedtuple(
    "H_option",
    [
        "save_mem",
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
    def __init__(self):
        """
        All the HCN's and HDN's should be in the same level.
        """
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
