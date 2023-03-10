# ==========================
#  ====== H structure =======
# ==========================

from rkgb.utils import *
from rkgb.Ptools import P_graph, P_node
from rkgb.Ktools import K_graph, K_C_node, K_D_node
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
    def __init__(self, name, main_target):
        self.name = name
        self.main_target = main_target
        self.deps = set()  # HCN set
        self.users = set()  # HCN set
        self.mem = 0
        self.kdn = None  # temporary attribute


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
        self.fwd_inputs = set()  # HDN set # -> inputs' data
        self.fwd_outputs = set()  # HDN set # -> outputs' data
        self.bwd_inputs = set()  # HDN set # -> outputs' grad
        self.bwd_outputs = set()  # HDN set # -> inputs' grad
        self.loss_hcn = None  #  HCN
        self.all_kcn_inside = set()  # temporary attribute

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


# TODO TODO add the compiling info
def P_and_K_to_H(pg: P_graph, kg: K_graph):
    #  -> This function is recursive in 'pg'
    hg = H_graph(f"Hg_{pg.graph_id}")

    # ** useful dicts **
    dict_hdn_to_kdn = dict()
    dict_kcn_to_hcn = dict()
    #  -> /!\ all kcn inside hdn, at any depth level /!\

    #  ** small functions to extract the info in kg **
    dict_kn = kg.dict_kn
    has_bwd = lambda mt: f"bwd_{mt}" in dict_kn
    has_phantoms = lambda mt: f"{mt} phantoms" in dict_kn
    get_kcn_fwd = lambda mt: dict_kn[f"fwd_{mt}"]
    get_kcn_bwd = lambda mt: dict_kn[f"bwd_{mt}"]
    get_kdn_data = lambda mt: dict_kn[f"{mt} data"]
    get_kdn_grad = lambda mt: dict_kn[f"{mt} grad"]
    get_kdn_phantoms = lambda mt: dict_kn[f"{mt} phantoms"]

    # ==* First, build the H_nodes *==
    for pn in pg.list_nodes:
        if pn.is_leaf:
            # ** Bottom level **
            mt = pn.main_target
            hcn_fwd = H_C_node(f"Fwd_{pn.name}")
            hdn_data = H_D_node(f"Data_bottom_level_{mt}", mt)
            kcn_fwd = get_kcn_fwd(mt)
            kdn_data = get_kdn_data(mt)
            hcn_fwd.fwd_time = kcn_fwd.time
            hcn_fwd.fwd_overhead = kcn_fwd.overhead
            hdn_data.mem = kdn_data.mem
            hdn_data.kdn = kdn_data
            hcns = [hcn_fwd]
            hdns = [hdn_data]
            #  ** bwd part **
            if has_bwd(mt):
                hcn_bwd = H_C_node(f"Bwd_{pn.name}")
                hdn_grad = H_D_node(f"Grad_bottom_level_{mt}", mt)
                kcn_bwd = get_kcn_bwd(mt)
                kdn_grad = get_kdn_grad(mt)
                hcn_bwd.is_fwd = False
                hcn_bwd.fwd_time = kcn_bwd.time
                hcn_bwd.fwd_overhead = kcn_bwd.overhead
                hdn_grad.mem = kdn_grad.mem
                hdn_grad.kdn = kdn_grad
                hcns.append(hcn_bwd)
                hdns.append(hdn_grad)
                #  ** last level graph **
                sub_hg = H_graph(f"Hg_{pn.name}")
                # TODO TODO : add the bottom option
                # -> use get_kdn_phantoms
                for kdn in kcn_fwd.users:
                    if "phantom" in kdn.name:
                        # there should be at most one phantom
                        mem = kdn.mem
                hopt = H_option(
                    sub_hg,
                    op_list=[],
                    alive_list=[],
                    direct_info={
                        "fwd_time": hcn_fwd.time,
                        "bwd_time": hcn_bwd.time,
                        "mem": mem,
                        "fwd_overhead": hcn_fwd.overhead,
                        "bwd_overhead": hcn_bwd.overhead,
                        "dep_inputs": [
                            # TODO: if H_edges are done, should read from HDN
                            kdn.name
                            for kdn in kcn_fwd.deps_global
                            if kdn in kcn_bwd.deps_global
                        ],
                    },
                )
                sub_hg.list_opt = [hopt]
                hcn_fwd.sub_graph = hcn_bwd.sub_graph = sub_hg
            else:
                sub_hg = None
        else:
            #  ** Recursive **
            sub_hg = P_and_K_to_H(pn.subgraph, kg)
            hcn_fwd = H_C_node(f"Fwd_{pn.name}")
            hcn_bwd = H_C_node(f"Bwd_{pn.name}")
            hcn_fwd.sub_graph = hcn_bwd.sub_graph = sub_hg
            hcn_bwd.is_fwd = False

            # TODO : hcn_fwd.fwd_time
            # TODO : hcn_fwd.fwd_overhead
            # TODO : hcn_bwd.bwd_time
            # TODO : hcn_bwd.bwd_overhead
            hcn_fwd.fwd_time = sum(
                sub_hcn.fwd_time for sub_hcn in sub_hg.list_hcn
            )
            hcn_fwd.fwd_overhead = sum(
                sub_hdn.mem for sub_hdn in sub_hg.list_hdn  # if not interfaces
            )
            # fwd_time and overhead are for fast forward so bwd node has none

            hcns = [hcn_fwd, hcn_bwd]
            hdns = []
            for hdn_output_data_in_sub_hg in sub_hg.fwd_outputs:
                mt = hdn_output_data_in_sub_hg.main_target
                mem = hdn_output_data_in_sub_hg.mem
                hdn_data = H_D_node(f"Data_{mt}_in_{sub_hg.name}", mt)
                hdn_data.mem = mem
                hdns.append(hdn_data)
            for hdn_output_grad_in_sub_hg in sub_hg.bwd_inputs:
                mt = hdn_output_grad_in_sub_hg.main_target
                mem = hdn_output_grad_in_sub_hg.mem
                hdn_grad = H_D_node(f"Grad_{mt}_in_{sub_hg.name}", mt)
                hdn_grad.mem = mem
                hdns.append(hdn_grad)

        # * register everything *
        for hn in hcns + hdns:
            hg.dict_hn[hn.name] = hn
        hg.list_hcn.extend(hcns)
        hg.list_hdn.extend(hdns)
        if not (sub_hg is None):
            hg.dict_hg[sub_hg.name] = sub_hg

    #  ===* Second, build the edges *===

    #  /!\ build hg.all_kcn_inside


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
    Info needed for HILP:
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

    def __init__(self, hgraph, op_list, alive_list, direct_info={}):
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
        if direct_info:
            self.mem = direct_info["mem"]
            self.fwd_time = direct_info["fwd_time"]
            self.bwd_time = direct_info["bwd_time"]
            self.fwd_overhead = direct_info["fwd_overhead"]
            self.bwd_overhead = direct_info["bwd_overhead"]
            self.dep_inputs = direct_info["dep_inputs"]
        else:
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
                        if (
                            hdn in hgraph.fwd_inputs
                            and hdn.name in self.dep_inputs
                        ):
                            self.dep_inputs.append(hdn.name)


# def find_bkcn(fkcn, kg):
#     assert fkcn.is_fwd
#     if fkcn.name.replace("fwd", "bwd") in kg.dict_kn:
#         bkcn = kg.dict_kn[fkcn.name.replace("fwd", "bwd")]
#         return bkcn
#     else:
#         return False


# def kcn_to_hopt(fkcn, bkcn):
#     mem = 0
#     for kdn in fkcn.users:
#         if "phantom" in kdn.name:  # there should be at most one phantom
#             mem = kdn.mem
#     h_opt = H_option(
#         H_graph(fkcn.name.strip("fwd")),
#         op_list=[],
#         alive_list=[],
#         direct_info={
#             "fwd_time": fkcn.time,
#             "bwd_time": bkcn.time,
#             "mem": mem,
#             "fwd_overhead": fkcn.overhead,
#             "bwd_overhead": bkcn.overhead,
#             "dep_inputs": [
#                 kdn.name for kdn in fkcn.deps_global if kdn in bkcn.deps_global
#             ],
#         },
#     )
#     return h_opt


# def get_dict_opt(kg):
#     dict_opt = {}
#     for kcn in kg.list_kcn:
#         if kcn.is_fwd:
#             bkcn = find_bkcn(kcn, kg)
#             if bkcn:
#                 h_opt = kcn_to_hopt(kcn, bkcn)
#                 dict_opt[h_opt.name] = h_opt
