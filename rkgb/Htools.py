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
        self.main_target = None  # Bottom level
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

    def sort_list_hcn(self):
        # -> copy paste K_graph.sort_list_kcn
        leaves_hcn = set()
        for hcn in self.list_hcn:
            if not hcn.is_fwd and len(hcn.users) == 0:
                leaves_hcn.add(hcn)
        root_hdn = H_D_node("", "")
        root_hcn = H_C_node("")
        root_hdn.deps = leaves_hcn
        root_hcn.deps = self.inputs_hdn_grad  # Not enought
        root_hcn.deps.add(root_hdn)
        self.list_hcn = l = shared_methods.sort_based_on_deps(root_hcn)
        l.remove(root_hcn)


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
            hdn_data = H_D_node(f"Data_bottom_level_{mt}", mt)
            kcn_fwd = get_kcn_fwd(mt)
            kdn_data = get_kdn_data(mt)
            dict_kcn_to_hcn[kcn_fwd] = hcn_fwd
            dict_hdn_to_kdn[hdn_data] = kdn_data
            hcn_fwd.is_leaf = True
            hcn_fwd.main_target = mt
            hcn_fwd.fwd_time = kcn_fwd.time
            hcn_fwd.fwd_overhead = kcn_fwd.overhead
            hdn_data.mem = kdn_data.mem
            hcns = [hcn_fwd]
            hdns = [hdn_data]
            #  ** bwd part **
            if has_bwd(mt):
                hcn_bwd = H_C_node(f"Bwd_{pn.name}")
                hdn_grad = H_D_node(f"Grad_bottom_level_{mt}", mt)
                kcn_bwd = get_kcn_bwd(mt)
                kdn_grad = get_kdn_grad(mt)
                dict_kcn_to_hcn[kcn_bwd] = hcn_bwd
                dict_hdn_to_kdn[hdn_grad] = kdn_grad
                hcn_bwd.is_leaf = True
                hcn_bwd.is_fwd = False
                hcn_bwd.main_target = mt
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
                    h_sched=H_sched([], [], {}),
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
                    hdn_io = H_D_node(f"Grad_{mt}_in_{hg.name}", mt)
                    hdn_io.is_data = False
                else:
                    hdn_io = H_D_node(f"Data_{mt}_in_{hg.name}", mt)
                hdn_io.mem = mem
                hdns.append(hdn_io)
                dict_hdn_to_kdn[hdn_io] = kdn_io

        # * register everything *
        for hn in hcns + hdns:
            hg.dict_hn[hn.name] = hn
        hg.list_hcn.extend(hcns)
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
        inputs_data.add(hdn_data)
        inputs_grad.add(hdn_grad)
    hg.interfaces = (
        hg.inputs_hdn_data
        | hg.inputs_hdn_grad
        | hg.outputs_hdn_data
        | hg.outputs_hdn_grad
    )

    #  =* loss_hcn *=
    hg.loss_hcn = loss_hcn = H_C_node(f"Loss_hcn_of_{hg.name}")
    hg.loss_hcn.main_target = "loss"
    hg.list_hcn.append(hg.loss_hcn)

    hg.list_hdn = list(dict_hdn_to_kdn.keys())

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

    # ** special input "source" **
    if "src" in pg.input_targets:
        hdn_data = dict_mt_to_hdn_data["src"]
        kdn_data = dict_hdn_to_kdn[hdn_data]
        for kcn in kdn_data.users_global:
            hcn = dict_kcn_to_hcn[kcn]
            hdn_data.users.add(hcn)
            hcn.deps.add(hdn_data)
        hdn_grad = dict_mt_to_hdn_grad["src"]
        kdn_grad = dict_hdn_to_kdn[hdn_grad]
        for kcn in kdn_grad.deps_global:
            hcn = dict_kcn_to_hcn[kcn]
            hdn_grad.deps.add(hcn)
            hcn.users.add(hdn_grad)
    hg.sort_list_hcn()

    return hg


# ==========================


# ==========================
# === printing functions ===
# ==========================

color_hcn_fwd = "blue"
color_hcn_bwd = "blueviolet"
color_special = "green"
color_hdn = "olive"
color_edge = "black"


def get_color(hn):
    if hn.main_target == "loss":
        return color_special
    if isinstance(hn, H_D_node):
        return color_hdn
    if hn.is_fwd:
        return color_hcn_fwd
    return color_hcn_bwd


def print_H_graph(hg: H_graph, name=None, open=True, render_format="svg"):
    # ----- init -----
    print(
        f"Hierarchical graph : \n"
        f"{len(hg.list_hcn)} H_C_nodes,\n"
        f"{len(hg.list_hdn)} H_D_nodes"
    )
    if name is None:
        name = "Hierarchical_graph"
    dot = graphviz.Digraph(name, comment="H_graph = Hierarchical graph")
    # ----- Core -----
    # * nodes *
    def print_hcn(hcn: H_C_node):
        mt = hcn.main_target
        dot.node(
            hcn.name,
            hcn.name,
            color=get_color(hcn),
            tooltip=(
                f"Fast Forward Time : {hcn.fwd_time}"
                f"Fast Forward Memory Overhead : "
                f"{irotor.MemSize(hcn.fwd_overhead)}"
            )
            if hcn.is_fwd
            else "",
        )

    def print_hdn(kdn):
        dot.node(
            kdn.name,
            kdn.name,
            color=get_color(kdn),
            tooltip=f"Mem {irotor.MemSize(kdn.mem)}",
        )

    for hcn in hg.list_hcn:
        print_hcn(hcn)
    for hdn in hg.list_hdn:
        print_hdn(hdn)

    # * edges *
    for hcn in hg.list_hcn:
        for req_hdn in hcn.deps:
            dot.edge(req_hdn.name, hcn.name, color=color_edge)
        for user_hdn in hcn.users:
            dot.edge(hcn.name, user_hdn.name, color=color_edge)

    #  ----- render -----
    small_fcts.graph_render(dot, open, "H", render_format)


# ***********
# * H_op *
# ***********


class H_op:
    def __init__(self, name, h_obj, is_fwd=True, is_del=False):
        self.name = name
        self.is_fwd = is_fwd
        self.is_del = is_del
        self.obj = h_obj
        if is_del:
            self.time = 0
            self.overhead = 0
        else:
            self.time = h_obj.fwd_time if is_fwd else h_obj.bwd_time
            self.overhead = h_obj.fwd_overhead if is_fwd else h_obj.bwd_overhead


class H_sched:
    def __init__(self, op_list: list, alive_list: list, sizes: dict):
        self.op_list = op_list
        self.alive_list = alive_list
        self.sizes = sizes
        self.ignore = []

    def alive_mem(self, i, ignore=[]):
        ignore += self.ignore
        alive_status = self.alive_list[i]
        return sum(
            self.sizes[k][v]
            for k, v in alive_status.items()
            if (v > -1 and k not in ignore)
        )

    def del_op(self, i):
        # this function does not correct alive status
        self.op_list[i] = ...  # TODO: create an empty op or placeholder

    def update_alive_list(self, target_status, start=0, end=-1):
        for alive_status in self.alive_list[start:end]:
            alive_status.update(target_status)

    # def keep_hdn(self, hdn_name, status=1, start=0, end=-1):
    #     # manually set the status of one hdn
    #     self.update_alive_list({hdn_name:status}, start, end)

    def extend(self, next_sched):
        common_k = self.sizes.keys() & next_sched.sizes.keys()
        current_end_status = self.alive_list[-1].copy()
        next_start_status = next_sched.alive_list[0].copy()

        self.op_list += next_sched.op_list
        self.sizes.update(next_sched.sizes)
        self.alive_list += next_sched.alive_list

        for k in common_k:
            del next_start_status[k]
        self.update_alive_list(
            next_start_status, start=0, end=len(self.alive_list) + 1
        )

        for k in common_k:
            del current_end_status[k]
        self.update_alive_list(
            current_end_status, start=len(self.alive_list) + 1, end=-1
        )

    def split_sched(self, split_idx):
        sched_0 = H_sched(
            self.op_list[: split_idx + 1],
            self.alive_list[: split_idx + 1],
            self.sizes,
        )
        sched_1 = H_sched(
            self.op_list[split_idx + 1 :],
            self.alive_list[split_idx + 1 :],
            self.sizes,
        )
        return sched_0, sched_1

    def collapse(self, i, sub_op_sched):
        # replace the i-th operation by the lower level op_sched
        self.op_list = (
            self.op_list[:i] + sub_op_sched.op_list + self.op_list[i + 1 :]
        )


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

    def __init__(self, hgraph, h_sched, direct_info={}):
        # when op_list and alive_list are empty, all the information can be
        # assigned directly
        self.h_sched = h_sched
        self.op_list = h_sched.op_list
        self.alive_list = h_sched.alive_list
        for i, op in enumerate(self.op_list):
            if "Loss" in op.name:
                self.loss_idx = i
                break
        L = len(self.op_list)
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

        for i, (op, alive_status) in enumerate(
            zip(self.op_list, self.alive_list)
        ):
            self.save_mem[i] = _sum_mem(alive_status, interfaces_names)
            if not op.is_del:
                self.time[i] = op.time
                self.overhead[i] = op.overhead

                # if op.is_fwd:
                #     self.time[i] = op.fwd_time
                #     self.overhead[i] = op.fwd_overhead
                # else:
                #     self.time[i] = op.bwd_time
                #     self.overhead[i] = op.bwd_overhead
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
            for op in self.op_list[self.loss_idx + 1 :]:
                if not op.is_del:
                    for hdn in hgraph.dict_hn[op.name].deps:
                        if (
                            hdn in hgraph.inputs_hdn_data
                            and hdn.name in self.dep_inputs
                        ):
                            self.dep_inputs.append(hdn.name)

        self.phantoms = set()
        for k, v in self.alive_list[self.loss_idx].items():
            if v > -1 and not k in interfaces_names:
                if k in hgraph.dict_hn:
                    self.phantoms.add(hgraph.dict_hn[k])
                elif k in hgraph.dict_hg:
                    self.phantoms.add(hgraph.dict_hg[k])
                else:
                    raise Warning(f"cannot find {k} in Hgraph")


def get_save_all_option(hgraph):
    def _can_del(i, hdn):
        for hcn in hdn.users:
            if hgraph.list_hcn.index(hcn) > i:
                return False
        return True

    op_list = []
    alive_list = []
    alive_status = {}
    sizes = {}
    for hdn in hgraph.list_hdn:
        if hdn not in hgraph.interfaces:
            alive_status[hdn.name] = (
                0 if (hdn in hgraph.inputs_hdn_data) else -1
            )
            sizes[hdn.name] = [hdn.mem]

    for hcn in hgraph.list_hcn:
        if hcn.is_fwd and hcn.sub_graph is not None:
            sub_g = hcn.sub_graph
            if not sub_g.list_opt:
                sub_opt = get_save_all_option(sub_g)
                sub_g.list_opt.append(sub_opt)
            alive_status[sub_g.name] = -1
            sizes[sub_g.name] = [h_opt.mem for h_opt in sub_g.list_opt]

    for i, hcn in enumerate(hgraph.list_hcn):
        if hcn.sub_graph is None:
            for hdn in hcn.users:
                alive_status[hdn.name] = 0
            op_list.append(
                H_op(hcn.name, h_obj=hcn, is_fwd=hcn.is_fwd, is_del=False)
            )
            alive_list.append(alive_status.copy())

            continue

        h_obj = hcn.sub_graph.list_opt[0]
        for hdn in hcn.users:
            alive_status[hdn.name] = 0
        if hcn.is_fwd:
            alive_status[hcn.sub_graph.name] = 0
        op_list.append(
            H_op(hcn.name, h_obj=h_obj, is_fwd=hcn.is_fwd, is_del=False)
        )
        alive_list.append(alive_status.copy())

        for hdn_name, alive in alive_status.items():
            if "Hg" in hdn_name:
                continue
            hdn = hgraph.dict_hn[hdn_name]
            if alive > -1 and _can_del(i, hdn):
                op_list.append(H_op("Del_" + hdn.name, hdn, is_del=True))
                alive_status[hdn_name] = -1
                alive_list.append(alive_status.copy())

        if not hcn.is_fwd:
            alive_status[hcn.sub_graph.name] = -1
            op_list.append(
                H_op("Del_" + hcn.sub_graph.name, h_obj, is_del=True)
            )
            alive_list.append(alive_status.copy())

    h_option = H_option(hgraph, H_sched(op_list, alive_list, sizes))
    return h_option

def replace(op_list, i, sub_op_list):
    # replace the i-th operation by the lower level op_sched
    op_list = (
        op_list[:i] + sub_op_list + op_list[i + 1 :]
    )

def collapse(op):
    if isinstance(op.obj, H_option) and op.obj.op_list:
        if "Del" in op.name:
            op_list_ = []
            for obj in op.obj.phantoms:
                if isinstance(obj, H_D_node):
                    op_list_.append(H_op(f"Del_{obj.name}", obj, is_del=True))
                else:
                    op_list_ += collapse(H_op(f"Del_{obj.name}", obj, is_del=True))

            return op_list_
        else:
            op_list_ = []
            if op.is_fwd:
                for sub_op in op.obj.op_list[:op.obj.loss_idx+1]:
                    op_list_ += collapse(sub_op)
            else:
                for sub_op in op.obj.op_list[op.obj.loss_idx+1:]:
                    op_list_ += collapse(sub_op)
            return op_list_
            
    else:
        return [op]
    

def get_bottom_op_list(op_list):
    bottom_op_list = []
    for i, op in enumerate(op_list):
        bottom_op_list += collapse(op)
    return bottom_op_list

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
