# This file is not actually used after updating rk_rotor.py
# ==========================
# definition file of RK_Chain
# also contains RK_Chain builder -> depends on use_chk.py
#  based on rotor/algorithms/parameters.py
# ==========================
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from ..main import get_sched, translate
from ...op_schedule import OpSchedule, DeleteOp, ComputeOp

import math
from copy import deepcopy

# ==========================
# ======== RK Block ========
# ==========================

class RK_Schedule(OpSchedule):
    # A special case of schedule, where only the forward is useful
    def __init__(self, op_list, cluster):
        super().__init__(op_list, loss_idx=len(op_list) - 1, cluster=cluster, correct_overhead=False)
        ## TODO: maybe accessor functions?
        # self.time = self.fwd_time
        # self.save = self.save_mem

def get_sched_peak_with_interfaces(sched, only_start=False, only_end=False):
    assert not (only_start and only_end)
    start_idx = sched.loss_idx if only_end else None
    end_idx = 1+sched.loss_idx if only_start else None
    return max(sched.save_mem_with_interfaces[start_idx:end_idx] + sched.overhead[start_idx:end_idx])


class RK_Block_Solution:
    # A solution contains a fwd_sched and a bwd_sched
    # and also size_a_bar, time_fwd, time_bwd, overhead_fwd, overhead_bwd

    # prologue_op_list is added to the forward part of op_sched. Can be []
    # pred_disable_from_bwd is a predicate for ops that should be disabled from the backward schedule.
    #                       Used to disable delete operations about the input data
    def __init__(self, block, sub_cluster, op_sched, prologue_op_list, pred_disable_from_bwd):
        # fwd_op_list = h_compute_node.sub_cluster.translate_op_list(
        fwd_op_list = translate(sub_cluster,
                                prologue_op_list + op_sched.op_list[: op_sched.loss_idx]
                                )  # + [Op(K_C_node("loss"))]
        bwd_op_list = translate(sub_cluster,
                                op_sched.op_list[op_sched.loss_idx + 1 :]
                                )  # start with loss op
        for op in bwd_op_list:
            if pred_disable_from_bwd(op):
                op.disabled = True

        self.fwd_sched = RK_Schedule(fwd_op_list, sub_cluster)
      
        self.bwd_sched = OpSchedule(
            bwd_op_list,
            loss_idx=0,
            cluster=sub_cluster,
            correct_overhead=False,
        ) 

        full_sched = OpSchedule(
            fwd_op_list
            + [ComputeOp(sub_cluster.loss_cnode)]
            + bwd_op_list,
            loss_idx=len(fwd_op_list) - 1,
            cluster=sub_cluster,
            correct_overhead=False,
        )  # different from op_sched because it contains the prologue if there is one
        ## TODO: why do I need full_sched? To compute save_mem and overhead?
        # self.bwd_sched.time = full_sched.bwd_time
        # self.bwd_sched.overhead = full_sched.bwd_overhead
        # self.bwd_sched.save = full_sched.save_mem[full_sched.loss_idx :]
        # Include in abar size the size of output Tensors which are alive when computing loss
        self.size_a_bar = full_sched.mem + sum(anode.mem for anode in full_sched.interfaces["output_data_anodes"]
                                               if full_sched.alive_list[full_sched.loss_idx][anode.name])
        assert (block.mem_output == sum(anode.mem for anode in full_sched.interfaces["output_data_anodes"]
                                        if full_sched.alive_list[full_sched.loss_idx][anode.name]))
        self.time_fwd = full_sched.fwd_time
        self.time_bwd = full_sched.bwd_time
        self.overhead_fwd = full_sched.fwd_overhead
        # for rotor, overhead needs to be equal to peak - size of input and output.
        #     for backward, input is [input + grad of output + output with phantoms] and output is [grad of input]
        self.overhead_fwd = get_sched_peak_with_interfaces(full_sched, only_start=True) - block.mem_input - self.size_a_bar
        self.overhead_bwd = get_sched_peak_with_interfaces(full_sched, only_end=True) - 2 * block.mem_input - self.size_a_bar - block.mem_output

class RK_Block:
    ## Build a block from a Compute node, or several compute nodes merged
    ## if the first ones have no BWD.
    def __init__(self, h_compute_nodes):
        assert len(h_compute_nodes) >= 1
        previous_nodes, final_node = h_compute_nodes[:-1], h_compute_nodes[-1]
        assert all(hcn.sub_cluster is None for hcn in previous_nodes)
        
        prologue_op_list = []
        for f_hcn in previous_nodes:
            prologue_op_list += f_hcn.ff_op_list

        self.Fn_sched = RK_Schedule(deepcopy(prologue_op_list) + final_node.ff_op_list, final_node.sub_cluster)

        # To build FC_sched, we need to remove delete operations from the prologue
        # Since we have a sequential structure, we assume only the first compute node of the prologue has inputs
        first_hcn = h_compute_nodes[0]
        input_anode_data = list(han.anode for han in first_hcn.deps)
        output_anode_data = list(han.anode for han in final_node.users)
        self.mem_input = sum(anode.mem for anode in input_anode_data)
        self.mem_output = sum(anode.mem for anode in output_anode_data)

        for op in prologue_op_list:
            if isinstance(op, DeleteOp) and op.target.anode in input_anode_data:
                print("Rotor, making Forward Check schedule: disabling deletion of input data",
                      op.target.anode.name)
                op.disabled = True
                ## TODO: understand why we "disable" it instead of removing it from the list
                ## like ff_op_list = [ x for x in prologue_op_list if not (isinstance(op, DeleteOp) and op.target.anode in input_anode_data]
                ## This would probably remove the need for deepcopy() above

        # TODO: is the .copy() necessary?
        self.Fc_sched = RK_Schedule(prologue_op_list.copy() + final_node.ff_op_list, final_node.sub_cluster)

        def is_delete_of_input_data(op):
            # By default, force bwd not to delete input data/grad
            return (isinstance(op, DeleteOp) and 
                    any(op.target.anode.main_target == anode.main_target for anode in input_anode_data))
            ## TODO: is there a difference between test on previous line and op.target.anode in input_anode_data that is used above?
            ##       Yes, main_target means also consider gradients

        self.sols = []
        for op_sched in get_sched(final_node.sub_cluster):
            solution = RK_Block_Solution(self, final_node.sub_cluster, op_sched, prologue_op_list, pred_disable_from_bwd=is_delete_of_input_data)
            self.sols.append(solution)
            
        self.overhead_fast_fwd = get_sched_peak_with_interfaces(self.Fc_sched) - self.mem_input - self.mem_output
        self.time_fast_fwd = self.Fc_sched.fwd_time

# ==========================
# ======== RK CHAIN ========
# ==========================

def make_list_of_empty_lists(n):
    return [[] for _ in range(n)]

class RK_Chain:
    def __init__(self, hg: HierarchicalGraph, mem_unit=1):
        self.body = []
        self.mem_unit = mem_unit
        loss_idx = hg.list_HCNs.index(hg.loss_hcn)

        no_grad_hcns = []
        for i, hcn in enumerate(hg.list_HCNs[:loss_idx]):  # toposorted
            # WARNING: we only make a Block if hcn has a BWD, otherwise we keep it to merge with the next one
            no_grad_hcns.append(hcn)
            if hcn.sub_cluster is not None:
                self.body.append(RK_Block(no_grad_hcns))
                no_grad_hcns = []
        

        mkl = make_list_of_empty_lists
        self.ln = len(self.body)
        nb = self.ln + 1
        self.fw = mkl(nb)
        self.bw = mkl(nb)
        self.cw = [None] * (nb + 1)
        self.cbw = mkl(nb + 1)
        self.fwd_tmp = mkl(nb)
        self.bwd_tmp = mkl(nb)
        self.ff_fwd_tmp = [None] * (nb)
        self.ff_fw = [None] * (nb)

        self.nb_sol = []
        for i, b in enumerate(self.body):
            self.nb_sol.append(len(b.sols))
            if self.nb_sol[-1] == 0:
                raise Exception(
                    f"We need at least one solution per block. "
                    f"Here {b.block_name} has no solution"
                )
            for sol in b.sols:
                self.fw[i].append(sol.time_fwd)
                self.bw[i].append(sol.time_bwd)
                self.cbw[i + 1].append(sol.size_a_bar)
                self.fwd_tmp[i].append(sol.overhead_fwd)
                self.bwd_tmp[i].append(sol.overhead_bwd)
            self.cw[i] = b.mem_input
            self.ff_fwd_tmp[i] = b.overhead_fast_fwd
            self.ff_fw[i] = b.time_fast_fwd

        self.cw[self.ln] = sum(value.mem for value in hg.output_data_HANs)  # the final output(s)

        self.cbw[-2] = [(c - self.cw[self.ln]) for c in self.cbw[-2]]

        # for the Loss block :
        self.nb_sol.append(1)
        self.fw[-1] = [0]
        self.bw[-1] = [0]
        self.cw[-1] = 0
        self.cbw[-1] = [0]
        self.fwd_tmp[-1] = [0]
        self.bwd_tmp[-1] = [0]
        self.ff_fwd_tmp[-1] = 0
        self.ff_fw[-1] = 0

        if mem_unit != 1:
            self.discretize()


    def discretize(self):
        def discretize_(values):
            return [math.ceil(value / self.mem_unit) for value in values]

        self.cw = discretize_(self.cw)
        self.cbw = [discretize_(x) for x in self.cbw]
        self.fwd_tmp = [discretize_(x) for x in self.fwd_tmp]
        self.bwd_tmp = [discretize_(x) for x in self.bwd_tmp]
        self.ff_fwd_tmp = discretize_(self.ff_fwd_tmp)

    def simulate(self, sequence, display=True, stopAtLoss=False):
        return sequence.simulate(self, display, stopAtLoss)
