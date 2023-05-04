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
        self.ff_op_list = []
        self.number = -1
        #  When toposorting the HCNs we want to respect the same order
        # as the one found for KCNs, ie follow variable number
        # (get_num_tar) AND add some special deps for artifacts
        #  To make it easy, we give a number to each hcn based on K_graph's
        # toposort : hcn.is_leaf -> hcn.number = kcn index in list_kcn
        # else -> hcn.number = min (sub_hcn.number in hcn.subgraph)


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
        self.list_sched = []
        self.inputs_hdn_data = set()  # HDN set -> inputs' data
        self.outputs_hdn_data = set()  # HDN set -> outputs' data
        self.outputs_hdn_grad = set()  # HDN set -> outputs' grad
        self.inputs_hdn_grad = set()  # HDN set -> inputs' grad

        self.loss_hcn = None  #  HCN
        self.all_kcn_inside = set()  # temporary attribute

    def add_sched(self, sched):
        pareto = True
        for opt in self.list_sched:
            if (
                opt.fwd_time + opt.bwd_time < sched.fwd_time + sched.bwd_time
            ) and (opt.mem < sched.mem):
                # should consider other factors like req_inputs
                pareto = False
            if (
                opt.fwd_time + opt.bwd_time > sched.fwd_time + sched.bwd_time
            ) and (opt.mem > sched.mem):
                self.list_sched.remove(opt)
        if pareto:
            self.list_sched.append(sched)
        # self.refine_scheds()

    def refine_scheds(self, expect_num=10):
        def exclude_one():
            def worse_sched(sched1, sched2):
                if (
                    sched1.fwd_time + sched1.bwd_time
                    > sched2.fwd_time + sched2.bwd_time
                ):
                    return sched1
                else:
                    return sched2

            # find the pair to remove one
            diff = 100
            for i, _sched1 in enumerate(self.list_sched[1:]):
                for _sched2 in self.list_sched[i + 1 :]:

                    if (
                        abs((_sched2.mem - _sched1.mem + 1) / (_sched2.mem + 1))
                        < diff
                    ):
                        sched1 = _sched1
                        sched2 = _sched2
                        diff = abs(
                            (_sched2.mem - _sched1.mem + 1) / (_sched2.mem + 1)
                        )
            # assert sched1 in self.list_sched
            # assert sched2 in self.list_sched
            self.list_sched.remove(worse_sched(sched1, sched2))

        while len(self.list_sched) > expect_num:
            # print(len(self.list_sched))
            exclude_one()

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
        op_list = []
        alive_mem = []
        alive_list = []
        alive_status = np.zeros(len(self.list_hdn), dtype=bool)
        for hdn in self.inputs_hdn_data:
            alive_status[self.list_hdn.index(hdn)] = 1
        loss_idx = self.list_hcn.index(self.loss_hcn)
        for i, hcn in enumerate(self.list_hcn[:loss_idx]):
            op_list += hcn.ff_op_list
            for hdn in hcn.users:
                alive_status[self.list_hdn.index(hdn)] = 1
            alive_list.append(alive_status.copy())
            alive_mem.append(
                sum(
                    hdn.mem
                    for j, hdn in enumerate(self.list_hdn)  # No phantom in FF
                    if alive_status[j]
                )
            )

            for j, hdn in enumerate(self.list_hdn):
                # if False:  # TODO: if hdn is interface
                # if hdn in self.inputs_hdn_data:
                #     continue
                if alive_status[j] and _can_del(i, hdn):
                    alive_status[j] = 0
                    # op_list.append(H_op(hdn.name, hdn, is_del=True))
                    op_list.append(Op(hdn.kdn))
                    alive_list.append(alive_status.copy())
                    alive_mem.append(
                        sum(
                            hdn.mem
                            for j, hdn in enumerate(self.list_hdn)
                            if alive_status[j]
                        )
                    )
        return max(alive_mem) - alive_mem[-1], op_list

    def sort_list_hcn(self):
        # -> copy paste K_graph.sort_list_kcn
        l1 = list(self.list_hcn)
        leaves_hcn = set()
        for hcn in self.list_hcn:
            if not hcn.is_fwd:
                if len(hcn.users) == 0:
                    leaves_hcn.add(hcn)
                else:
                    really_useful = False
                    for user_hdn in hcn.users:
                        if len(user_hdn.users) != 0:
                            really_useful = True
                    if not really_useful:
                        leaves_hcn.add(hcn)
        root_hdn = H_D_node("", "")
        root_hcn = H_C_node("")
        root_hdn.deps = leaves_hcn
        root_hcn.deps = set(self.inputs_hdn_grad)  # Not enough
        root_hcn.deps.add(root_hdn)
        self.list_hcn = l = RK_sort_based_on_deps(root_hcn)
        l.remove(root_hcn)
        if set(l1) != set(l):
            print(f"BEFORE : {len(l1)}")
            print([hn.name for hn in l1])
            print(f"AFTER : {len(l)}")
            print([hn.name for hn in l])
            print("loss users : ")
            print([hn.name for hn in self.loss_hcn.users])
            print("outputs_hdn_data", self.outputs_hdn_data)
            print("outputs_hdn_grad", self.outputs_hdn_grad)


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
    has_grad = lambda mt: f"{mt} grad" in dict_kn
    has_phantoms = lambda mt: f"{mt} phantoms" in dict_kn
    get_kcn_fwd = lambda mt: dict_kn[f"fwd_{mt}"]
    get_kcn_bwd = lambda mt: dict_kn[f"bwd_{mt}"]
    get_kdn_data = lambda mt: dict_kn[f"{mt} data"]
    get_kdn_grad = lambda mt: dict_kn[f"{mt} grad"]
    get_kdn_phantoms = lambda mt: dict_kn[f"{mt} phantoms"]
    # get_kdn_data = lambda mt: (
    # dict_kn[f"{mt} data"]
    # if f"{mt} data" in dict_kn
    # else K_D_node())
    # get_kdn_grad = lambda mt: (
    # dict_kn[f"{mt} grad"]
    # if f"{mt} grad" in dict_kn
    # else K_D_node())
    # get_kdn_phantoms = lambda mt: (
    # dict_kn[f"{mt} phantoms"]
    # if f"{mt} phantoms" in dict_kn
    # else K_D_node())

    # ==* First, build the H_nodes *==
    for pn in pg.list_nodes:
        if pn.is_leaf:
            # ** Bottom level **
            mt = pn.main_target
            hcn_fwd = H_C_node(f"Fwd_{mt}")
            hdn_data = H_D_node(f"Data_{mt}", mt)
            kcn_fwd = get_kcn_fwd(mt)
            kdn_data = get_kdn_data(mt)
            dict_kcn_to_hcn[kcn_fwd] = hcn_fwd
            dict_hdn_to_kdn[hdn_data] = kdn_data
            hcn_fwd.is_leaf = True
            hcn_fwd.main_target = mt
            hcn_fwd.number = kg.list_kcn.index(kcn_fwd)
            hcn_fwd.fwd_time = kcn_fwd.time
            hcn_fwd.fwd_overhead = kcn_fwd.overhead
            hcn_fwd.ff_op_list = [
                # H_op(hcn_fwd.name, hcn_fwd, is_fwd=True, is_del=False)
                Op(kcn_fwd, fast_forward=True)
            ]
            hdn_data.mem = kdn_data.mem
            hcns = [hcn_fwd]
            hdns = [hdn_data]
            #  ** bwd part **
            if has_bwd(mt):
                hcn_bwd = H_C_node(f"Bwd_{mt}")
                hdn_grad = H_D_node(f"Grad_{mt}", mt)
                kcn_bwd = get_kcn_bwd(mt)
                kdn_grad = get_kdn_grad(mt)
                dict_kcn_to_hcn[kcn_bwd] = hcn_bwd
                dict_hdn_to_kdn[hdn_grad] = kdn_grad
                hcn_bwd.is_leaf = True
                hcn_bwd.is_fwd = False
                hcn_bwd.main_target = mt
                hcn_bwd.number = kg.list_kcn.index(kcn_bwd)
                # hcn_bwd.fwd_time = kcn_bwd.time
                # hcn_bwd.fwd_overhead = kcn_bwd.overhead
                hdn_grad.is_data = False
                hdn_grad.mem = kdn_grad.mem
                hcns.append(hcn_bwd)
                hdns.append(hdn_grad)
                #  ** last level graph **
                sub_hg = H_graph(f"Hg_{mt}")
                #  -> Build the bottom option
                if has_phantoms(mt):
                    ph_mem = get_kdn_phantoms(mt).mem
                else:
                    ph_mem = 0
                h_sched = H_sched([], [], {}, sub_hg)
                direct_info = {
                    "fwd_time": kcn_fwd.time,
                    "bwd_time": kcn_bwd.time,
                    "mem": ph_mem,
                    "fwd_overhead": kcn_fwd.overhead,
                    "bwd_overhead": kcn_bwd.overhead,
                    "dep_interfaces_data": [
                        # TODO: should read from HDN
                        # kdn.name
                        # for kdn in kcn_fwd.deps_global
                        # if kdn in kcn_bwd.deps_global
                    ],
                }
                for k, v in direct_info.items():
                    setattr(h_sched, k, v)
                # sub_hg.list_sched = [h_sched]
                sub_hg.list_sched = []

                # hopt = H_option(
                #     sub_hg,
                #     h_sched=H_sched([], [], {}),
                #     direct_info={
                #         "fwd_time": kcn_fwd.time,
                #         "bwd_time": kcn_bwd.time,
                #         "mem": ph_mem,
                #         "fwd_overhead": kcn_fwd.overhead,
                #         "bwd_overhead": kcn_bwd.overhead,
                #         "dep_interfaces_data": [
                #             # TODO: should read from HDN
                #             # kdn.name
                #             # for kdn in kcn_fwd.deps_global
                #             # if kdn in kcn_bwd.deps_global
                #         ],
                #     },
                # )
                # sub_hg.list_sched = [hopt]
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
            # number :
            nb_min_fwd = 9999999
            nb_min_bwd = 9999999
            for sub_hcn in sub_hg.list_hcn:
                if sub_hcn.is_fwd:
                    nb_min_fwd = min(sub_hcn.number, nb_min_fwd)
                else:
                    nb_min_bwd = min(sub_hcn.number, nb_min_bwd)
            hcn_fwd.number = nb_min_fwd
            hcn_bwd.number = nb_min_bwd
            hcn_fwd.fwd_time = sum(
                sub_hcn.fwd_time for sub_hcn in sub_hg.list_hcn
            )
            (
                hcn_fwd.fwd_overhead,
                hcn_fwd.ff_op_list,
            ) = sub_hg.fast_fwd_overhead()
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
                    hdn_io = H_D_node(f"Grad_in_{hg.name}_{mt}", mt)
                    hdn_io.is_data = False
                else:
                    hdn_io = H_D_node(f"Data_in_{hg.name}_{mt}", mt)
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
        hdn_data = H_D_node(f"Data_{inp_mt}_in_{hg.name}", inp_mt)
        hdn_data.mem = kdn_data.mem
        dict_mt_to_hdn_data[inp_mt] = hdn_data
        dict_hdn_to_kdn[hdn_data] = kdn_data
        inputs_data.add(hdn_data)
        hg.list_hdn.append(hdn_data)
        hg.dict_hn[hdn_data.name] = hdn_data
        if has_grad(inp_mt):
            kdn_grad = get_kdn_grad(inp_mt)
            hdn_grad = H_D_node(f"Grad_{inp_mt}_in_{hg.name}", inp_mt)
            hdn_grad.is_data = False
            hdn_grad.mem = kdn_grad.mem
            dict_mt_to_hdn_grad[inp_mt] = hdn_grad
            dict_hdn_to_kdn[hdn_grad] = kdn_grad
            inputs_grad.add(hdn_grad)
            hg.list_hdn.append(hdn_grad)
            hg.dict_hn[hdn_grad.name] = hdn_grad
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
                    if not (hcn in hdn.deps):
                        hdn.users.add(hcn)
                        hcn.deps.add(hdn)

    for hdn in hg.outputs_hdn_data:
        hdn.users.add(loss_hcn)
        loss_hcn.deps.add(hdn)
    for hdn in hg.outputs_hdn_grad:
        hdn.deps.add(loss_hcn)
        loss_hcn.users.add(hdn)

    # ** special input "source" **
    if "sources" in pg.input_targets:
        hdn_data = dict_mt_to_hdn_data["sources"]
        kdn_data = dict_hdn_to_kdn[hdn_data]
        for kcn in kdn_data.users_global:
            if kcn in dict_kcn_to_hcn:
                hcn = dict_kcn_to_hcn[kcn]
                hdn_data.users.add(hcn)
                hcn.deps.add(hdn_data)
        hdn_grad = dict_mt_to_hdn_grad["sources"]
        kdn_grad = dict_hdn_to_kdn[hdn_grad]
        if kdn_grad.deps_global == set():
            hg.list_hdn.remove(hdn_grad)
        else:
            for kcn in kdn_grad.deps_global:
                if kcn in dict_kcn_to_hcn:
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
    """
        The operation types:
        1. Run HCN: fast forward (thus HCN must be forward)
        2. Del HDN: delete HDN
        3. Run H_sched/H_option: run the corresponding fwd/bwd sched
        4. Del H_sched/H_option: delete the phantoms passed from fwd to bwd
        """

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


class Op:
    def __init__(self, kn, fast_forward=False, disabled=False, detach=True):
        self.kn = kn
        self.name = kn.name
        self.fast_forward = fast_forward
        self.disabled = disabled
        self.detach = detach
        self.is_del = isinstance(kn, K_D_node)

    def __repr__(self):
        return "Disabled" * self.disabled + self.name


class H_sched:
    """
    The H_sched correspond to one fwd/bwd schedule. When op_list is empty, 
    it represents the bottom level fwd/bwd and information is assigned directly.

    """

    def __init__(
        self, op_list: list, alive_list: list, sizes: dict, hgraph: H_graph
    ):
        self.op_list = op_list
        self.op_name_list = [op.name for op in op_list]
        self.alive_list = alive_list
        self.sizes = sizes
        self.hgraph = hgraph
        self.ignore = []
        self.fwd_overhead_correction = []
        self.bwd_overhead_correction = []
        self.bottom_op_list = []

    def alive_mem(self, i, ignore=[]):
        ignore += self.ignore
        alive_status = self.alive_list[i]
        return sum(
            self.sizes[k][v]
            for k, v in alive_status.items()
            if (v > -1 and k not in ignore)
        )

    def get_info(self):
        """
        A function that compute the time/memory information based on 
        op_list and alive_list. If they are updated, should run get_info() again.
        """
        hgraph = self.hgraph
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
                    mem += hgraph.dict_hg[k].list_sched[v].mem if v > -1 else 0
            return mem

        def get_overhead_(save, overhead):
            return max(save + overhead) - save[-1]

        self.interface_names = [hdn.name for hdn in hgraph.interfaces]

        for i, (op, alive_status) in enumerate(
            zip(self.op_list, self.alive_list)
        ):
            self.save_mem[i] = _sum_mem(alive_status, self.interface_names)
            if not op.is_del:
                self.time[i] = op.time
                self.overhead[i] = op.overhead

        if self.op_list:
            self.mem = self.save_mem[self.loss_idx]
            self.fwd_time = np.sum(self.time[: self.loss_idx + 1])
            self.bwd_time = np.sum(self.time[self.loss_idx + 1 :])
            # self.fwd_overhead = get_overhead_(
            #     self.save_mem[: self.loss_idx + 1],
            #     self.overhead[: self.loss_idx + 1],
            # )
            # self.bwd_overhead = get_overhead_(
            #     self.save_mem[self.loss_idx + 1 :],
            #     self.overhead[self.loss_idx + 1 :],
            # )

            # # In practice, h_sched is assumed to be still alive after BWD
            # # this will lead to negative overhead, but it makes
            # # bwd_overhead[i]+save[i]=real memory usage

            self.fwd_overhead = (
                max(self.overhead[: self.loss_idx + 1]) - self.mem
            )
            self.bwd_overhead = (
                max(self.overhead[self.loss_idx + 1 :]) - self.mem
            )

            # names of additional HDNs that are required by BWD
            self.dep_interfaces_data = set()
            for i, op in enumerate(self.op_list[self.loss_idx + 1 :]):
                if not op.is_del:
                    for hdn in hgraph.dict_hn[op.name].deps:
                        if hdn in hgraph.inputs_hdn_data:
                            self.dep_interfaces_data.add(hdn.kdn.name)
                        if hdn in hgraph.outputs_hdn_data:
                            if hdn.deps and (
                                list(hdn.deps)[0].name
                                not in self.op_name_list[self.loss_idx + 1 :][
                                    :i
                                ]  # if not generated during bwd
                            ):
                                self.dep_interfaces_data.add(hdn.kdn.name)

            self.phantoms = set()
            for k, v in self.alive_list[self.loss_idx].items():
                if v > -1 and not k in self.interface_names:
                    if k in hgraph.dict_hn:
                        self.phantoms.add(hgraph.dict_hn[k])
                    elif k in hgraph.dict_hg:
                        # delete phantom of a H_sched
                        self.phantoms.add(hgraph.dict_hg[k].list_sched[v])
                    else:
                        raise Warning(f"cannot find {k} in Hgraph")

            self.correct_overhead()

    def correct_overhead(self):
        # correction terms of overhead, each term represents one step in op_list

        interfaces_status = []
        for hdn in self.hgraph.inputs_hdn_data:  # Input of Fwd
            interfaces_status.append((hdn.name, self.loss_idx))  # After fwd
            if hdn.kdn.name in self.dep_interfaces_data:
                interfaces_status.append(
                    (hdn.name, len(self.op_list))
                )  # After Bwd
        for hdn in self.hgraph.outputs_hdn_data:  # Output of Fwd
            interfaces_status.append((hdn.name, 0))  # Before fwd?
            if hdn.kdn.name in self.dep_interfaces_data:
                # TODO: add the case when output is generated then deleted during bwd

                interfaces_status.append(
                    (hdn.name, len(self.op_list))
                )  # After Bwd
            else:
                interfaces_status.append((hdn.name, -1))  # After Bwd

        for hdn in self.hgraph.outputs_hdn_grad:
            interfaces_status.append((hdn.name, len(self.op_list)))  # After Bwd
        for hdn in self.hgraph.inputs_hdn_grad:
            interfaces_status.append(
                (hdn.name, self.loss_idx + 1)
            )  # Before Bwd

        for i, (op, alive_status) in enumerate(
            zip(self.op_list, self.alive_list)
        ):
            if i == self.loss_idx:
                continue
            correction_term = {
                "save": self.save_mem[i],
                "overhead": self.overhead[i],
            }
            for (hdn_name, index) in interfaces_status:
                hdn = self.hgraph.dict_hn[hdn_name]
                if index == -1:
                    # special case: output_data in BWD without dependency
                    # If outside is alive, no need to correct;
                    # Otherwise, add hdn to memory
                    if i > self.loss_idx and alive_status[hdn_name] > -1:
                        correction_term["save"] += hdn.mem
                        correction_term[(hdn.kdn.name, False)] = -hdn.mem
                    continue

                if (
                    alive_status[hdn_name] > -1
                    or (index > self.loss_idx) != (i > self.loss_idx)
                    # or not hdn_name
                ):
                    # interfaces_status is useful when:
                    # 1. hdn is not alive
                    # 2. Fwd to Fwd, Bwd to Bwd
                    continue

                if i >= index:  # if exist before
                    if (  # and not deleted in between
                        f"Del_{hdn_name}"
                        not in self.op_name_list[index : i + 1]
                    ):
                        correction_term[(hdn.kdn.name, True)] = -hdn.mem
                    else:
                        correction_term[(hdn.kdn.name, "always")] = -hdn.mem
                else:  # if exist afterwards
                    if not (hdn in self.hgraph.outputs_hdn_data) and (
                        hdn.deps
                        and (
                            list(hdn.deps)[0].name
                            in self.op_name_list[i : index + 1]
                        )
                    ):  # and not generated in between
                        # check if output_data is created after i
                        correction_term[(hdn.kdn.name, False)] = -hdn.mem
                    else:
                        correction_term[(hdn.kdn.name, "always")] = -hdn.mem

            if (
                i < self.loss_idx
                and correction_term not in self.fwd_overhead_correction
            ):
                self.fwd_overhead_correction.append(correction_term)
            elif (
                i > self.loss_idx
                and correction_term not in self.bwd_overhead_correction
            ):
                self.bwd_overhead_correction.append(correction_term)

        def refine_correction(correction):
            # Some correction terms are not useful because they will not be peak
            min_overhead = max(
                sum(correction_term.values()) for correction_term in correction
            )
            for i, correction_term in enumerate(correction):
                if (
                    correction_term["save"] + correction_term["overhead"]
                    < min_overhead
                ):
                    # This step will not be the peak even without correction
                    correction.pop(i)

        refine_correction(self.fwd_overhead_correction)
        refine_correction(self.bwd_overhead_correction)

        # for correction in self.fwd_overhead_correction:

    def get_bottom_op_list(self):
        def collapse_op(op):
            # if isinstance(op.obj, H_C_node):
            if hasattr(op.obj, "sub_graph"):
                if "Loss" in op.obj.name:
                    return [op]
                return op.obj.ff_op_list
            # elif isinstance(op.obj, H_sched):
            elif hasattr(op.obj, "op_list"):
                if not op.obj.op_list:
                    # bottom level
                    if op.is_del:
                        hgraph = op.obj.hgraph
                        return [H_op(f"Del_{hgraph.name}", hgraph, is_del=True)]
                    else:
                        return [op]
                else:
                    if "Del" in op.name:
                        op_list_ = []
                        for obj in op.obj.phantoms:
                            # if isinstance(obj, H_D_node):
                            if hasattr(obj, "kdn"):
                                op_list_.append(
                                    H_op(f"Del_{obj.name}", obj, is_del=True)
                                )
                            else:
                                op_list_ += collapse(
                                    H_op(f"Del_sub_graph", obj, is_del=True)
                                )

                        return op_list_
                    else:
                        op_list_ = []
                        if op.is_fwd:
                            for sub_op in op.obj.op_list[: op.obj.loss_idx + 1]:
                                op_list_ += collapse(sub_op)
                        else:
                            for sub_op in op.obj.op_list[op.obj.loss_idx + 1 :]:
                                op_list_ += collapse(sub_op)
                        return op_list_

            else:
                return [op]

        if not self.bottom_op_list:
            for i, op in enumerate(self.op_list):
                self.bottom_op_list += collapse_op(op)
        return self.bottom_op_list


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
        # if hdn not in hgraph.interfaces:
        alive_status[hdn.name] = 0 if (hdn in hgraph.inputs_hdn_data) else -1
        sizes[hdn.name] = [hdn.mem]

    for hcn in hgraph.list_hcn:
        if hcn.is_fwd and hcn.sub_graph is not None:
            sub_g = hcn.sub_graph
            if not sub_g.list_sched:
                sub_opt = get_save_all_option(sub_g)
                sub_g.add_sched(sub_opt)
            alive_status[sub_g.name] = -1
            sizes[sub_g.name] = [h_sched.mem for h_sched in sub_g.list_sched]

    for i, hcn in enumerate(hgraph.list_hcn):
        if hcn.sub_graph is None:
            for hdn in hcn.users:
                alive_status[hdn.name] = 0
            op_list.append(
                H_op(hcn.name, h_obj=hcn, is_fwd=hcn.is_fwd, is_del=False)
            )
            alive_list.append(alive_status.copy())

            continue

        h_obj = hcn.sub_graph.list_sched[0]
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

    h_sched = H_sched(op_list, alive_list, sizes, hgraph)
    h_sched.get_info()
    return h_sched


def replace(op_list, i, sub_op_list):
    # replace the i-th operation by the lower level op_sched
    op_list = op_list[:i] + sub_op_list + op_list[i + 1 :]


def collapse(op):
    # if isinstance(op.obj, H_C_node):
    if hasattr(op.obj, "sub_graph"):
        if "Loss" in op.obj.name:
            return [op]
        return op.obj.ff_op_list
    # elif isinstance(op.obj, H_sched):
    elif hasattr(op.obj, "op_list"):
        if not op.obj.op_list:
            # bottom level
            if op.is_del:
                hgraph = op.obj.hgraph
                return [H_op(f"Del_{hgraph.name}", hgraph, is_del=True)]
            else:
                return [op]
        else:
            if "Del" in op.name:
                op_list_ = []
                for obj in op.obj.phantoms:
                    # if isinstance(obj, H_D_node):
                    if hasattr(obj, "kdn"):
                        op_list_.append(
                            H_op(f"Del_{obj.name}", obj, is_del=True)
                        )
                    else:
                        op_list_ += collapse(
                            H_op(f"Del_sub_graph", obj, is_del=True)
                        )

                return op_list_
            else:
                op_list_ = []
                if op.is_fwd:
                    for sub_op in op.obj.op_list[: op.obj.loss_idx + 1]:
                        op_list_ += collapse(sub_op)
                else:
                    for sub_op in op.obj.op_list[op.obj.loss_idx + 1 :]:
                        op_list_ += collapse(sub_op)
                return op_list_

    else:
        return [op]


def get_bottom_op_list(op_list):
    bottom_op_list = []
    for i, op in enumerate(op_list):
        bottom_op_list += collapse(op)
    return bottom_op_list


def get_kop_list(bottom_op_list, hg: H_graph):
    def find_main_target(op_name, targets):
        for target in targets:
            if target in op_name:
                return target
        return None

    def fake_kop(name):
        return K_op(K_C_node(target=name), disabled=True)

    kn_list = []
    targets = set()
    dict_kn = {}
    for kcn in hg.all_kcn_inside:
        dict_kn[kcn.name] = kcn
        targets.add(kcn.main_target)
        for kdn in kcn.users:
            dict_kn[kdn.name] = kdn
        for kdn in kcn.deps_real:
            dict_kn[kdn.name] = kdn

    for op in bottom_op_list:
        if "Loss" in op.name:
            # kn_list.append(op.name)
            kn_list.append(fake_kop(op.name))
            continue
        mt = find_main_target(op.name, targets)
        if mt is None:
            # kn_list.append(op.name)
            kn_list.append(fake_kop(op.name))
            continue
        if op.is_del:
            if "Data" in op.name:
                kn_list.append(K_op(dict_kn[f"{mt} data"]))
            elif "Grad" in op.name:
                kn_list.append(K_op(dict_kn[f"{mt} grad"]))
            elif "Hg" in op.name and f"{mt} phantoms" in dict_kn:
                kn_list.append(K_op(dict_kn[f"{mt} phantoms"]))
            else:
                kn_list.append(fake_kop(op.name))
                # kn_list.append(op.name)

        else:
            if op.is_fwd:
                kn_list.append(
                    K_op(dict_kn[f"fwd_{mt}"], ff=hasattr(op.obj, "ff_op_list"))
                )
            else:
                kn_list.append(K_op(dict_kn[f"bwd_{mt}"]))
    return kn_list


def get_kn_list(op_list, kg):
    def find_main_target(op_name, targets):
        for target in targets:
            if target in op_name:
                return target
        return None

    kn_list = []
    targets = set()
    for kcn in kg.list_kcn:
        targets.add(kcn.main_target)

    for op in op_list:
        if "Loss" in op.name:
            # kn_list.append(op.name)
            kn_list.append(kg.dict_kn[f"fwd_loss"])
            continue
        mt = find_main_target(op.name, targets)
        if mt is None:
            kn_list.append(op.name)
            continue
        if op.is_del:
            if "Data" in op.name:
                kn_list.append(kg.dict_kn[f"{mt} data"])
            elif "Grad" in op.name:
                kn_list.append(kg.dict_kn[f"{mt} grad"])
            elif "Hg" in op.name and f"{mt} phantoms" in kg.dict_kn:
                kn_list.append(kg.dict_kn[f"{mt} phantoms"])
            else:
                kn_list.append(op.name)

        else:
            if op.is_fwd:
                kn_list.append(kg.dict_kn[f"fwd_{mt}"])
            else:
                kn_list.append(kg.dict_kn[f"bwd_{mt}"])
    return kn_list


def refine_kn_list(kn_list):
    refined_kn_list = []
    for kn in kn_list:
        if not isinstance(kn, str):
            refined_kn_list.append(kn)
        else:
            refined_kn_list.append(kn)

    return refined_kn_list


def print_collapse(op_list):
    for i, op in enumerate(op_list):
        print(i, op.name)
        if len(collapse(op)) > 1:
            for sub_op in collapse(op):
                print(f"| {sub_op.name}")


def find_graph(target, hg):
    for sub_name, sub_g in hg.dict_hg.items():
        for hcn in sub_g.list_hcn:
            if target in hcn.name and hcn.is_fwd:
                print(sub_name)
