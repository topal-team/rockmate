from rkgb.Htools import H_cluster
from .twremat_utils import *
from .op_schedule import OpSchedule as New_OpSchedule
from .op_schedule import Op
from .main import Solver, get_cluster_budget


class TwRemat(Solver):
    class Config:
        def __init__(
            self,
            contains_data_node=False,
            allow_loss_recomputation=False,
            mem_unit=1024**2,
        ):
            self.mem_unit = mem_unit
            self.contains_data_node = contains_data_node
            self.allow_loss_recomputation = allow_loss_recomputation

    def __init__(self, config=None):
        super().__init__(config)

    def _can_solve(self, cluster: H_cluster):
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
            if isinstance(cluster, H_cluster):
                list_sched.append(self.solve_cluster(cluster, int(budget))[1])

        return list_sched

    def solve_cluster(self, cluster, budget):
        node_info, target, loss = get_twremat_graph(
            cluster,
            self.config.contains_data_node,
            self.config.allow_loss_recomputation,
        )
        steps = runtwremat(node_info, budget, target, loss)

        kcn_id_to_node = {cnode.unique_id: cnode for cnode in cluster.list_kcn}
        data_nodes = cluster.list_kdn

        op_list = []
        op_name_list = []

        for i, step in enumerate(steps):
            step_type, cnode_id = step

            cnode_id = int(cnode_id)
            cnode = kcn_id_to_node[cnode_id]

            if step_type == "compute":
                runOp = Op(cnode, detach=True, disabled=("loss" in cnode.name))
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
                    delOp = Op(data_nodes[idx])
                    op_list.append(delOp)
                    op_name_list.append(delOp.name)

        sched = New_OpSchedule(op_list, cluster=cluster)
        sched.solver = "TWRemat"
        return op_name_list, sched
