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
        self.is_leaf = False


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
        self.is_data = True  #  otherwise, grad
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
        self.inputs_hdn_data = set()  # HDN set -> inputs' data
        self.outputs_hdn_data = set()  # HDN set -> outputs' data
        self.outputs_hdn_grad = set()  # HDN set -> outputs' grad
        self.inputs_hdn_grad = set()  # HDN set -> inputs' grad
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

    def fast_fwd_overhead(self):
        def _can_del(i, hdn):
            for hcn in hdn.users:
                if not hcn.is_fwd:
                    continue
                if self.list_hcn.index(hcn) > i:
                    return False
            return True

        loss_idx = self.list_hcn.index(self.loss_hcn)
        alive_mem = []
        alive_list = []
        alive_status = np.zeros(len(self.list_hdn), dtype=bool)
        loss_idx = self.list_hcn.index(self.loss_hcn)
        for i, hcn in enumerate(self.list_hcn[:loss_idx]):
            for hdn in hcn.users:
                alive_status[self.list_hdn.index(hdn)] = 1
            for j, hdn in enumerate(self.list_hdn):
                if False:  # TODO: if hdn is interface
                    continue
                if alive_status[j] and _can_del(i, hdn):
                    alive_status[j] = 0
            alive_list.append(alive_status.copy())
            alive_mem.append(
                sum(
                    hdn.mem
                    for j, hdn in enumerate(self.list_hdn)
                    if alive_status[j]
                )
            )
        return max(alive_mem) - alive_mem[-1]


def P_and_K_to_H(pg: P_graph, kg: K_graph):
    #  -> This function is recursive in 'pg'
    hg = H_graph(f"Hg_{pg.graph_id}")

    # ** useful dicts **
    dict_mt_to_hdn_data = dict()
    dict_mt_to_hdn_grad = dict()
    #  -> to easily find the inputs and outputs
    dict_hdn_to_kdn = dict()
    #  A hdn represents exactly one kdn
    dict_kcn_to_hcn = dict()
    #  At a fixed level of depth, a kcn can be found in only one hcn
    #  -> These two dict make it super easy to build edges

    #  ** small functions to extract K_nodes from kg **
    dict_kn = kg.dict_kn
    dict_kn[kg.input_kdn_data.name] = kg.input_kdn_data
    dict_kn[kg.input_kdn_grad.name] = kg.input_kdn_grad
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
            hcn_fwd.is_leaf = True
            hdn_data = H_D_node(f"Data_bottom_level_{mt}", mt)
            kcn_fwd = get_kcn_fwd(mt)
            kdn_data = get_kdn_data(mt)
            dict_kcn_to_hcn[kcn_fwd] = hcn_fwd
            dict_hdn_to_kdn[hdn_data] = kdn_data
            hcn_fwd.fwd_time = kcn_fwd.time
            hcn_fwd.fwd_overhead = kcn_fwd.overhead
            hdn_data.mem = kdn_data.mem
            hcns = [hcn_fwd]
            hdns = [hdn_data]
            #  ** bwd part **
            if has_bwd(mt):
                hcn_bwd = H_C_node(f"Bwd_{pn.name}")
                hcn_bwd.is_leaf = True
                hdn_grad = H_D_node(f"Grad_bottom_level_{mt}", mt)
                kcn_bwd = get_kcn_bwd(mt)
                kdn_grad = get_kdn_grad(mt)
                dict_kcn_to_hcn[kcn_bwd] = hcn_bwd
                dict_hdn_to_kdn[hdn_grad] = kdn_grad
                hcn_bwd.is_fwd = False
                # hcn_bwd.fwd_time = kcn_bwd.time
                # hcn_bwd.fwd_overhead = kcn_bwd.overhead
                hdn_grad.is_data = False
                hdn_grad.mem = kdn_grad.mem
                hcns.append(hcn_bwd)
                hdns.append(hdn_grad)
                #  ** last level graph **
                sub_hg = H_graph(f"Hg_{pn.name}")
                #  -> Build the bottom option
                if has_phantoms(mt):
                    ph_mem = get_kdn_phantoms(mt).mem
                else:
                    ph_mem = 0
                hopt = H_option(
                    sub_hg,
                    op_list=[],
                    alive_list=[],
                    direct_info={
                        "fwd_time": kcn_fwd.time,
                        "bwd_time": kcn_bwd.time,
                        "mem": ph_mem,
                        "fwd_overhead": kcn_fwd.overhead,
                        "bwd_overhead": kcn_bwd.overhead,
                        "dep_inputs": [
                            # TODO: should read from HDN
                            # kdn.name
                            # for kdn in kcn_fwd.deps_global
                            # if kdn in kcn_bwd.deps_global
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
            hcn_fwd.fwd_time = sum(
                sub_hcn.fwd_time for sub_hcn in sub_hg.list_hcn
            )
            hcn_fwd.fwd_overhead = sub_hg.fast_fwd_overhead()
            # fwd_time and overhead are for fast forward so bwd node has none

            for kcn in sub_hg.all_kcn_inside:
                dict_kcn_to_hcn[kcn] = hcn_fwd if kcn.is_fwd else hcn_bwd

            hcns = [hcn_fwd, hcn_bwd]
            hdns = []
            io_of_sub_hg = sub_hg.outputs_hdn_data.union(
                sub_hg.outputs_hdn_grad
            )  # We don't create the nodes for the inputs
            for hdn_io_in_sub_hg in io_of_sub_hg:
                mt = hdn_io_in_sub_hg.main_target
                mem = hdn_io_in_sub_hg.mem
                kdn_io = hdn_io_in_sub_hg.kdn
                if kdn_io.kdn_type == "grad":
                    hdn_io = H_D_node("Grad_{mt}_in_{hg.name}", mt)
                    hdn_io.is_data = False
                else:
                    hdn_io = H_D_node("Data_{mt}_in_{hg.name}", mt)
                hdn_io.mem = mem
                hdns.append(hdn_io)
                dict_hdn_to_kdn[hdn_io] = kdn_io

        # * register everything *
        for hn in hcns + hdns:
            hg.dict_hn[hn.name] = hn
        hg.list_hcn.extend(hcns)
        hg.list_hdn.extend(hdns)
        if not (sub_hg is None):
            hg.dict_hg[sub_hg.name] = sub_hg
        for hdn in hdns:
            if hdn.is_data:
                dict_mt_to_hdn_data[hdn.main_target] = hdn
            else:
                dict_mt_to_hdn_grad[hdn.main_target] = hdn

    #  ==* Inputs / Outputs *==
    hg.outputs_hdn_data = set(
        dict_mt_to_hdn_data[inp] for inp in pg.output_targets
    )
    hg.outputs_hdn_grad = set(
        dict_mt_to_hdn_grad[inp] for inp in pg.output_targets
    )
    hg.inputs_hdn_data = inputs_data = set()
    hg.inputs_hdn_grad = inputs_grad = set()
    for inp_mt in pg.input_targets:

        assert inp_mt not in dict_mt_to_hdn_data
        assert inp_mt not in dict_mt_to_hdn_grad
        kdn_data = get_kdn_data(inp_mt)
        kdn_grad = get_kdn_grad(inp_mt)
        hdn_data = H_D_node(f"Data_{inp_mt}_in_{hg.name}", inp_mt)
        hdn_grad = H_D_node(f"Grad_{inp_mt}_in_{hg.name}", inp_mt)
        hdn_grad.is_data = False
        hdn_data.mem = kdn_data.mem
        hdn_grad.mem = kdn_grad.mem
        dict_mt_to_hdn_data[inp_mt] = hdn_data
        dict_mt_to_hdn_grad[inp_mt] = hdn_grad
        dict_hdn_to_kdn[hdn_data] = kdn_data
        dict_hdn_to_kdn[hdn_grad] = kdn_grad
        inputs_data.add(kdn_data)
        inputs_grad.add(kdn_grad)

    #  =* loss_hcn *=
    hg.loss_hcn = loss_hcn = H_C_node(f"Loss_hcn_of_{hg.name}")
    hg.list_hcn.append(hg.loss_hcn)

    #  =* register hdn.kdn and hg.all_kcn_inside *=
    for hdn, kdn in dict_hdn_to_kdn.items():
        hdn.kdn = kdn
    hg.all_kcn_inside = set(dict_kcn_to_hcn.keys())

    #  ===* Second, build the edges *===
    for hdn in hg.list_hdn:
        kdn = dict_hdn_to_kdn[hdn]
        for kcn in kdn.deps:
            if kcn is not kg.loss_kcn:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    hdn.deps.add(hcn)
                    hcn.users.add(hdn)
        for kcn in kdn.users_real:
            if kcn is not kg.loss_kcn:
                if kcn in dict_kcn_to_hcn:
                    hcn = dict_kcn_to_hcn[kcn]
                    hdn.users.add(hcn)
                    hcn.deps.add(hdn)

    for hdn in hg.outputs_hdn_data:
        hdn.users.add(loss_hcn)
        loss_hcn.deps.add(hdn)
    for hdn in hg.outputs_hdn_grad:
        hdn.deps.add(loss_hcn)
        loss_hcn.users.add(hdn)

    return hg


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
            hgraph.inputs_hdn_data,
            hgraph.outputs_hdn_data,
            hgraph.outputs_hdn_grad,
            hgraph.inputs_hdn_grad,
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
