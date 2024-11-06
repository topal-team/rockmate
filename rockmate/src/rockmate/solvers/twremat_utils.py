from copy import deepcopy
import numpy as np
import torch

import rkgb
# from .def_op import RunOp, DelOp, OpSchedule

import subprocess
import os
import tempfile

HOMEDIR = os.path.expanduser("~")
TWREMAT = os.path.join(HOMEDIR, "twremat")


# Allow users to pass 'humanized' memlimit values as strings.
def parse_memlimit(memlimit):
    if memlimit[-1] == "K":
        return int(memlimit[:-1]) * 1024
    elif memlimit[-1] == "M":
        return int(memlimit[:-1]) * 1024**2
    elif memlimit[-1] == "G":
        return int(memlimit[:-1]) * 1024**3
    else:
        return int(memlimit)


def runtwremat(gr, memlimit, target, loss, verbose=False):
    if type(memlimit) is str:
        memlimit = parse_memlimit(memlimit)

    fname = tempfile.mktemp()
    outname = tempfile.mktemp()

    with open(fname, "w") as fp:
        print("p remat2", file=fp)

        if memlimit != None:
            print(f"memlimit {memlimit}", file=fp)

        for n, info in gr.items():
            deps = " ".join(str(d) for d in info["deps"])

            if info["type"] == "normal":
                cpu = info["cpu"]
                mem = info["mem"]
                overhead_value = info["overhead"]
                weight = f"cpu {cpu} mem {mem} overhead {overhead_value}"
            elif info["type"] == "effectful":
                weight = "effectful"
            elif info["type"] == "pointer":
                weight = "pointer"
            if n in target:
                tstr = "target"
            else:
                tstr = ""

            if n in loss:
                loss_str = "loss"
            else:
                loss_str = ""

            print(f"node {n} deps {deps} {weight} {tstr} {loss_str}", file=fp)

    if verbose:
        print(" ".join([TWREMAT, fname, outname]))
    result = subprocess.run(["twremat", fname, outname], check=True, capture_output=not verbose)

    out = []
    with open(outname, "r") as fp:
        for line in fp:
            line = line.split()
            if line and line[0] == "c":
                out.append(("compute", int(line[1])))
            elif line and line[0] == "f":
                out.append(("free", int(line[1])))
            elif line:
                print(line)
                exit()
    return out


def get_rockmate_graphs(model, sample, device="cuda"):
    _model = deepcopy(model).to(device)
    for _, p in _model.named_parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    if isinstance(sample, list):
        _sample = []
        for input in sample:
            _sample.append(deepcopy(input).to(device))
    else:
        _sample = deepcopy(sample).to(device)

    # Get graphs with rk-GB
    try:
        rkgb_res = rkgb.rkgb.Result(model=_model,
                               model_args=_sample,
                               inspection_device=torch.device("cuda"))
        H_cluster = rkgb_res.hierarchical_cluster

        ### OLD RKGB START   

        # rkgb_res = rkgb.make_all_graphs(
        #    _model, _sample, check_device_is_gpu=True
        # )
        # # rkgb.print_all_graphs(rkgb_res,name="fno1d",render_format="pdf")
        # # forward_graph = rkgb_res.S_graph
        # K_graph = rkgb_res.K_graph

        # K_graph.fake_input_kdn_grad() ## FOR ORIGINAL ROCKMATE COMPATIBILITY

        ### OLD RKGB END

    except Exception as e:
        del _model
        del _sample
        print(e)
        raise Exception(
            "K-graph can't be built on one GPU for given model hyperparameters"
        )

    del _model
    del _sample

    #return K_graph
    return H_cluster


def get_twremat_graph(H_cluster, contains_data_node=False, allow_loss_recomputation=False
):
    """Builds fw-bckw graph, containing only computational nodes

    For each computational node `cnode` from K_graph collects:
    - memory info on input data tensors, `mem_inputs`
    - memory info on output data tensors, `mem_outputs`
    - memory info on computational overhead, `cnode.overhead`
    - set of parent computational nodes, `deps`

    Parameters
    ----------
    K_graph: rkgb.K_graph
        A data flow graph obrained with rk-GB, it has two types of nodes: data and computation nodes.

    Returns
    -------
        dict of dict
            Returns a dictionary `node_info`, where keys are ids of computational nodes and each value is a dictionary,
            which contains collected info on cnode in the format compatible with input to Haskell code of TWRemat, i.e.
        ::

            node_info[cnode.unique_id]= {
            'type': 'normal',
            'cpu': mem_inputs + mem_outputs,
            'mem': mem_outputs,
            'overhead': cnode.mem_overhead,
            'deps': cnode_deps
            }
        int
            Id of the computational node which is indicated as a final node (the node, which does not have any children nodes)
    """

    node_info = {}
    targets = []
    for cnode in H_cluster.list_cnodes:
        users_inside = [
            dnode
            for dnode in cnode.users
            if (dnode in H_cluster.list_anodes and not dnode in H_cluster.interfaces['input_grad_anodes'])
        ]
        if len(users_inside) == 0:
            targets.append(cnode.unique_id)

    for cnode in H_cluster.list_cnodes:
        mem_inputs = 0
        mem_outputs = 0
        cnode_deps = [
            cn.unique_id 
            for cn in cnode.deps_through_artifacts
            if cn in H_cluster.list_cnodes
        ]
        for dnode in cnode.deps_real:
            mem_inputs += dnode.mem

            for cn in H_cluster.list_cnodes:
                if dnode in cn.users:
                    cnode_deps.append(cn.unique_id)

        for dnode in cnode.deps_fake:
            mem_inputs += dnode.mem

            for cn in H_cluster.list_cnodes:
                if dnode in cn.users:
                    cnode_deps.append(cn.unique_id)

        # if (cnode.main_target != 'loss'):
        if not cnode.unique_id in targets:
            for dnode in cnode.users:
                mem_outputs += dnode.mem


        node_info[cnode.unique_id] = {
            "type": "normal",
            "cpu": mem_inputs + mem_outputs,
            "mem": mem_outputs,
            "overhead": cnode.mem_overhead if cnode.mem_overhead else 0,
            "deps": cnode_deps,
        }

    if not allow_loss_recomputation:
        loss_node = [
            [node for node in H_cluster.list_cnodes if "loss" in node.name][
                0
            ].unique_id
        ]
    else:
        loss_node = []

    if not targets:
        raise ValueError("no targets")
    if len(node_info) < 1:
        raise ValueError("no node_info")
    # if not loss_node:
    #     raise ValueError("no loss_node")
    return node_info, targets, loss_node

    # ### OLD RKGB START

    # # target = K_graph.list_kcn[-1].unique_id  # final computational node
    # targets = []
    # for kcn in K_graph.list_kcn:
    #     users_inside = [
    #         dn
    #         for dn in kcn.users
    #         if (dn in K_graph.list_kdn and not dn in K_graph.interfaces["inputs_kdn_grad"])
    #     ]
    #     if len(users_inside) == 0:
    #         targets.append(kcn.unique_id)

    # # for cnode in K_graph.list_kcn:
    # #     mem_inputs = 0
    # #     mem_outputs = 0
    # #     cnode_deps = []

    # #     for dnode in cnode.deps_real:
    # #         mem_inputs += dnode.mem

    # #         for cn in dnode.deps:
    # #             cnode_deps.append(cn.unique_id)

    # for cnode in K_graph.list_kcn:
    #     mem_inputs = 0
    #     mem_outputs = 0
    #     cnode_deps = [
    #         cn.unique_id
    #         for cn in cnode.deps_through_size_artifacts
    #         if cn in K_graph.list_kcn
    #     ]

    #     for dnode in cnode.deps_real:
    #         mem_inputs += dnode.mem

    #         for cn in K_graph.list_kcn:
    #             if dnode in cn.users:
    #                 cnode_deps.append(cn.unique_id)

    #     # for dnode in cnode.deps_fake:
    #     #     mem_inputs += dnode.mem
    #     #     for cn in dnode.deps:
    #     #         cnode_deps.append(cn.unique_id)

        # for dnode in cnode.deps_fake:
        #     mem_inputs += dnode.mem

        #     for cn in K_graph.list_kcn:
        #         if dnode in cn.users:
        #             cnode_deps.append(cn.unique_id)

        # # if (cnode.main_target != 'loss'):
        # if not cnode.unique_id in targets:
        #     for dnode in cnode.users:
        #         mem_outputs += dnode.mem

    #     node_info[cnode.unique_id] = {
    #         "type": "normal",
    #         "cpu": mem_inputs + mem_outputs,
    #         "mem": mem_outputs,
    #         "overhead": cnode.overhead if cnode.overhead else 0,
    #         "deps": cnode_deps,
    #     }

    # if not allow_loss_recomputation:
    #     loss_node = [
    #         [node for node in K_graph.list_kcn if "loss" in node.name][
    #             0
    #         ].unique_id
    #     ]
    # else:
    #     loss_node = []
    # if not targets:
    #     raise ValueError("no targets")
    # if len(node_info) < 1:
    #     raise ValueError("no node_info")
    # # if not loss_node:
    # #     raise ValueError("no loss_node")
    # return node_info, targets, loss_node
    
    # ### OLD RKGB END


def twremat_to_rockmate_schedule(K_graph, steps):
    kcn_id_to_node = {cnode.unique_id: cnode for cnode in K_graph.list_kcn}
    data_nodes = K_graph.list_kdn + [
        K_graph.input_kdn_grad,
        K_graph.input_kdn_data,
    ]

    print(f"OUTPUT SIZE TO KEEP ----{K_graph.output_kdn_data.mem}")

    op_list = []
    op_name_list = []
    alive_list = []
    alive_status = np.zeros(len(data_nodes), dtype=bool)
    alive_status[-1] = True  # input data tensor should be always alive!

    for i, step in enumerate(steps):
        step_type, cnode_id = step

        cnode_id = int(cnode_id)
        cnode = kcn_id_to_node[cnode_id]

        if step_type == "compute":
            ### Output data of the cnode computation should be stored
            for dnode in cnode.users:
                alive_status[data_nodes.index(dnode)] = 1
            alive_list.append(alive_status.copy())

            op = RunOp(cnode)
            op.no_grad = False
            op_list.append(op)
            op_name_list.append([op.name, op.op_type])

            # ### Input data to the cnode computation that won't be used by later computations should be deleted

            # # Get id of all computations that appear in the schedule after cnode
            # cnodes_compute_after = set([step[1] for step in list(filter(lambda step: step[0]=='compute', steps[i+1:]))])

            # # Get id of all data tensors that will serve as inputs to all computations performed after cnode
            # deps_cnodes_after = set()
            # for cn_id in cnodes_compute_after:
            #     cn = kcn_id_to_node[cn_id]
            #     for dn in set.union(cn.deps_real, cn.deps_fake):
            #     # for dn in set.union(cn.deps_real):
            #         deps_cnodes_after.add(dn.unique_id)
            # deps_cnodes_after.update([K_graph.input_kdn_data.unique_id])

            # # Free data tensors that served as inputs to cnode, but won't contribute to later computations
            # for dnode in data_nodes:
            #     idx = data_nodes.index(dnode)
            #     if alive_list[-1][idx] == 1 and not (dnode.unique_id in deps_cnodes_after):

            #         alive_status[idx] = 0
            #         alive_list.append(alive_status.copy())
            #         # print(alive_status)

            #         op = DelOp(data_nodes[idx])
            #         op.proxy = True
            #         op_list.append(op)
            #         op_name_list.append([op.name, op.op_type])

        elif step_type == "free":
            ### Twremat instruction "free computaion" we interpret as deleting data outputs of cnode computation
            for dnode in cnode.users:
                idx = data_nodes.index(dnode)
                if alive_status[idx]:
                    op = DelOp(data_nodes[idx])
                    op.proxy = True
                    op_list.append(op)
                    op_name_list.append([op.name, op.op_type])
                    alive_status[idx] = 0
                    alive_list.append(alive_status.copy())

        sched = OpSchedule(
            op_list,
            alive_list,
            K_graph.input_kdn_data,
            K_graph.input_kdn_grad,
            K_graph.output_kdn_data,
            K_graph.list_kdn,
            no_grad=False,
        )

    # from collections import Counter
    # counter = Counter(sched.op_name_list)
    # for k, v in counter.items():
    #     if 'fwd' in k:
    #         print(k, v, k.replace('fwd', 'bwd'), counter[k.replace('fwd', 'bwd')])

    # print(steps)
    # print(op_name_list)

    # print(f'Steps: {len(steps)},  op list: {len(sched.op_list)}')

    pred_mem = []
    for a, op in zip(sched.alive_list, sched.op_list):
        acc_mem = np.dot(a, sched.mem_sizes) - sched.input_size[1]

        pred_mem.append(acc_mem)
        if op.op_type == "Run":
            pred_mem[-1] += op.overhead

    print(pred_mem)
    print("peak_mem from op schedule " + str(np.max(np.array(pred_mem))))

    return sched
