from PIL import Image
from rkgb.Ktools import *  # aux_print_graph
from rkgb.Htools import *  # aux_print_graph
import graphviz

color_kcn_fwd = "blue"
color_kcn_bwd = "blueviolet"
color_special = "black"  # "green"
color_kdn = "blue"  # "olive"


def get_color(kn):
    if kn.name in color_dict:
        return color_dict[kn.name]
    if isinstance(kn, K_D_node):
        return color_kdn
    if kn.is_fwd:
        return "black"  # color_kcn_fwd
    return "black"  # color_kcn_bwd


def aux_print_graph(dot, kg, uniq_num):
    def uni(tar):
        return f"_{uniq_num}_{tar}"

    def node(i, l, **kwargs):
        dot.node(uni(i), l, **kwargs)

    def edge(i1, i2, **kwargs):
        dot.edge(uni(i1), uni(i2), **kwargs)

    # *** nodes ***
    def print_kcn(kcn):
        mt = kcn.main_target
        if mt == "loss":
            node(kcn.name, "LOSS KCN", color=color_special)
        else:
            # lbl = kcn.get_code() if kcn.is_fwd else f"backward of {mt}"
            lbl = f"forward of {mt}" if kcn.is_fwd else f"backward of {mt}"
            node(
                kcn.name,
                lbl,
                color=get_color(kcn),
                tooltip=(
                    f"Time : {kcn.time}\n" f"Mem overhead : {kcn.overhead}"
                ),
            )

    def print_kdn(kdn):
        node(
            kdn.name,
            kdn.name,
            color=get_color(kdn),
            shape="rect",
            tooltip=f"Mem {kdn.mem}",
        )

    for kcn in kg.list_kcn:
        print_kcn(kcn)
    for kdn in kg.list_kdn:
        print_kdn(kdn)

    # *** edges ***
    for kcn in kg.list_kcn:
        for req_kdn in kcn.deps_real:
            c = get_color(req_kdn)
            edge(req_kdn.name, kcn.name, color=c)
        for req_kdn in kcn.deps_fake:
            c = get_color(req_kdn)
            edge(req_kdn.name, kcn.name, color=c, style="dashed")
    for kdn in kg.list_kdn:
        for req_kcn in kdn.deps:
            edge(req_kcn.name, kdn.name, color=get_color(req_kcn))

    #  *** io - global relations ***
    inp_data = kg.input_kdn_data
    inp_grad = kg.input_kdn_grad
    kwargs = {"color": color_special, "style": "dashed"}
    node(inp_data.name, inp_data.name, **kwargs)
    node(inp_grad.name, inp_grad.name, **kwargs)
    inp_data_users_only_global = (
        inp_data.users_global - inp_data.users_real.union(inp_data.users_fake)
    )
    inp_grad_deps_only_global = inp_grad.deps_global - inp_grad.deps
    for user_inp_data in inp_data_users_only_global:
        edge(inp_data.name, user_inp_data.name, **kwargs)
    for req_inp_grad in inp_grad_deps_only_global:
        edge(req_inp_grad.name, inp_grad.name, **kwargs)


# images = []

# op_sched = newmod.fwd_seq.seq[1].op_sched
# for i,op in enumerate(op_sched.op_list):
#     color_dict = {}
#     for j,kdn_name in enumerate(op_sched.kdn_names):
#         color_dict[kdn_name] = color_kdn if op_sched.alive_list[i][j] else "grey"
#     if op.op_type == "Run": color_dict[op.name] = "red"
#     dot = graphviz.Digraph("kg")
#     aux_print_graph(dot, kg, 0)
#     dot.render(filename=f"{i}_op", directory="graphviz_dir", format='png')

# for i,op in enumerate(op_sched.op_list):
#     images.append(Image.open(f"graphviz_dir/{i}_op.png"))

# images[0].save('kg.gif',
#                save_all=True, append_images=images[1:],
#                optimize=False, duration=1000, loop=0)


#######################################


color_hcn_fwd = "blue"
color_hcn_bwd = "blueviolet"
color_hcn_fwd = "black"
color_hcn_bwd = "black"
color_special = "green"
color_hdn = "blue"
color_edge = "black"


def get_color(hn, color_dict):
    if hasattr(hn, "main_target") and hn.main_target == "loss":
        return color_special
    if hn.name in color_dict:
        return color_dict[hn.name]
    if isinstance(hn, H_D_node):
        return color_hdn
    if hn.is_fwd:
        return color_hcn_fwd
    return color_hcn_bwd


def print_H_graph(
    dot, hg: H_graph, color_dict, i, name=None, open=True, render_format="svg"
):
    # ----- init -----

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
            color=get_color(hcn, color_dict),
            tooltip=(
                f"Fast Forward Time : {hcn.fwd_time}"
                f"Fast Forward Memory Overhead : "
                f"{irotor.MemSize(hcn.fwd_overhead)}"
            )
            if hcn.is_fwd
            else "",
            penwidth="3",
        )

    def print_hdn(hdn):
        dot.node(
            hdn.name,
            hdn.name,
            color=get_color(hdn, color_dict),
            tooltip=f"Mem {irotor.MemSize(hdn.mem)}",
            shape="box",
            penwidth="3",
        )

    def print_phantom(hdn):
        dot.node(
            f"Phantoms {hdn.name}",
            f"Phantoms {hdn.name}",
            color=get_color(hdn, color_dict),
            shape="box",
            penwidth="3",
            # tooltip=f"Mem {irotor.MemSize(hdn.mem)}",
        )

    for hcn in hg.list_hcn:
        print_hcn(hcn)
    for hdn in hg.list_hdn:
        print_hdn(hdn)

    for hcn in hg.list_hcn:
        if hcn.sub_graph is not None and hcn.is_fwd:
            print_phantom(hcn.sub_graph)

    # * edges *
    for hcn in hg.list_hcn:
        for req_hdn in hcn.deps:
            dot.edge(req_hdn.name, hcn.name, color=color_edge)
        for user_hdn in hcn.users:
            dot.edge(hcn.name, user_hdn.name, color=color_edge)
        if hcn.sub_graph is not None and hcn.is_fwd:
            dot.edge(
                hcn.name, f"Phantoms {hcn.sub_graph.name}", color=color_edge
            )
            dot.edge(
                f"Phantoms {hcn.sub_graph.name}",
                hcn.name.replace("Fwd", "Bwd"),
                color=color_edge,
            )

    # print("lets print")
    dot.render(
        filename=f"{i}_op", directory="graphviz_dir/animate", format="png"
    )

    #  ----- render -----
    # small_fcts.graph_render(dot, open, "H", render_format)


def animated(op_sched, hg):
    print(
        f"Hierarchical graph : \n"
        f"{len(hg.list_hcn)} H_C_nodes,\n"
        f"{len(hg.list_hdn)} H_D_nodes"
    )
    images = []
    for i, op in enumerate(op_sched.op_list):
        color_dict = {}
        for j, hdn_name in enumerate(op_sched.sizes.keys()):
            color_dict[hdn_name] = (
                color_hdn if op_sched.alive_list[i][hdn_name] > -1 else "grey"
            )
        for hcn in hg.list_hcn:
            if op.name == hcn.name:
                color_dict[hcn.name] = "red"
            if hcn.sub_graph is not None and hcn.sub_graph.name == op.name:
                if hcn.is_fwd == op.is_fwd:
                    color_dict[hcn.name] = "red"
        # if i + 1 == len(op_sched.op_list) or not op_sched.op_list[i + 1].is_del:
        if not op.is_del:
            dot = graphviz.Digraph(
                "hg",
                node_attr={"fixedsize": "true", "width": "2", "height": "1.5"},
            )
            print_H_graph(dot, hg, color_dict, i)
            # if op.op_type == "Run":
            #     color_dict[op.name] = "red"

            # for i, op in enumerate(op_sched.op_list):
            # if not op.is_del:

            images.append(Image.open(f"graphviz_dir/animate/{i}_op.png"))

    images[0].save(
        "hg.gif",
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000,
        loop=0,
    )

