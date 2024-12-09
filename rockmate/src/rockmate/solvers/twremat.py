# from rkgb.Htools import H_cluster
from rkgb.core.hierarchical import HierarchicalCluster
from .twremat_utils import runtwremat, get_twremat_graph
from ..op_schedule import OpSchedule, ComputeOp, DeleteOp, Activation
from .main import Solver, get_cluster_budget
from dataclasses import dataclass


class TwRemat(Solver):
    '''This solver is a wrapper for the TwRemat treewidth-based rematerialization algorithm.

    The TwRemat algorithm is implemented in https://github.com/lemonlabsuk/ai-scala-developer
    and described in the paper

    Ravi Kumar, Manish Purohit, Zoya Svitkina, Erik Vee, Joshua Wang. Efficient Rematerialization
    for Deep Networks, NeurIPS
    2019. https://papers.nips.cc/paper/9653-efficient-rematerialization-for-deep-networks.pdf

    This class uses an external twremat executable that should be available in the PATH. A
    singularity image containing all requirements is linked in the project README.
    '''

    @dataclass
    class Config:
        mem_unit: int = 1024**2
        contains_data_node: bool = False
        allow_loss_recomputation: bool = False
        verbose: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _can_solve(self, cluster: HierarchicalCluster):
        return not cluster.is_bottom

    def solve(self, cluster, budgets=None):
        if not self._can_solve(cluster):
            return []
        if budgets is None:
            budgets = get_cluster_budget(
                cluster.representee_cluster, with_save_budget=False
            )
        else:
            budgets = budgets
        list_sched = []
        for budget in budgets:
            if isinstance(cluster, HierarchicalCluster):
                list_sched.append(self.solve_cluster(cluster, int(budget))[1])

        return list_sched

    def solve_cluster(self, cluster, budget):
        node_info, target, loss = get_twremat_graph(
            cluster,
            self.config.contains_data_node,
            self.config.allow_loss_recomputation,
        )
        steps = runtwremat(node_info, budget, target, loss, verbose=self.config.verbose)

        kcn_id_to_node = {cnode.unique_id: cnode for cnode in cluster.list_cnodes}
        data_nodes = cluster.list_anodes

        op_list = []
        op_name_list = []
        loss_idx = None

        for i, step in enumerate(steps):
            step_type, cnode_id = step

            cnode_id = int(cnode_id)
            cnode = kcn_id_to_node[cnode_id]

            if step_type == "compute":
                if "loss" in cnode.name and loss_idx is None: #for multiple loss nodes, we need the first one
                    loss_idx = len(op_list)

                runOp = ComputeOp(cnode, disabled=("loss" in cnode.name))
                op_list.append(runOp)
                op_name_list.append(runOp.name)

                # ### Input data to the cnode computation that won't be used by later computations should be deleted

                # # Get id of all computations that appear in the schedule after cnode
                # cnodes_compute_after = set([step[1] for step in list(filter(lambda step: step[0]=='compute', steps[i+1:]))])

                # # Get id of all data tensors that will serve as inputs to all computations performed after cnode
                # deps_cnodes_after = set()
                # for cn_id in cnodes_compute_after:
                #     cn = kcn_id_to_node[cn_id]
                #     # for dn in set.union(cn.deps_real, cn.deps_fake):
                #     for dn in set.union(cn.deps_real):
                #         deps_cnodes_after.add(dn.unique_id)
                # deps_cnodes_after.update([cluster.list_kdn.unique_id])

                # # Free data tensors that served as inputs to cnode, but won't contribute to later computations
                # for dnode in data_nodes:
                #     idx = data_nodes.index(dnode)
                #     if not (dnode.unique_id in deps_cnodes_after):
                #         op_list.append(Op(data_nodes[idx]))
                #         op_name_list.append(op.name)

            elif step_type == "free":
                ### Twremat instruction "free computaion" we interpret as deleting data outputs of cnode computation
                for dnode in cnode.users:
                    idx = data_nodes.index(dnode)
                    delOp = DeleteOp(Activation(data_nodes[idx]))
                    op_list.append(delOp)
                    op_name_list.append(delOp.name)

        sched = OpSchedule(op_list, cluster=cluster, loss_idx=loss_idx)
        sched.solver = "TWRemat"
        return op_name_list, sched
