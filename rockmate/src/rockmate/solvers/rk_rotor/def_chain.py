# This file is not actually used after updating rk_rotor.py
# ==========================
# definition file of RK_Chain
# also contains RK_Chain builder -> depends on use_chk.py
#  based on rotor/algorithms/parameters.py
# ==========================
from rkgb.core.hierarchical import HierarchicalGraph, HierarchicalCluster
from ..main import get_sched, translate
from ...op_schedule import OpSchedule, DeleteOp, ComputeOp
from . import def_sequence as rkseq

import math
from copy import deepcopy

# ==========================
# ======== RK Block ========
# ==========================


def forward_schedule(op_list, cluster):
    return OpSchedule(op_list, loss_idx=len(op_list) - 1, cluster=cluster, correct_overhead=False)
def backward_schedule(op_list, cluster):
    return OpSchedule(op_list, loss_idx=0, cluster=cluster, correct_overhead=False)

def get_sched_peak_with_interfaces(sched, only_start=False, only_end=False):
    assert not (only_start and only_end)
    start_idx = sched.loss_idx+1 if only_end else None
    end_idx = sched.loss_idx if only_start else None
    return max(sched.save_mem_with_interfaces[start_idx:end_idx] + sched.overhead[start_idx:end_idx])

def get_oplist_peak(op_list, cluster):
    sched = OpSchedule(op_list, loss_idx=len(op_list) - 1, cluster=cluster, correct_overhead=False)
    return max(sched.save_mem_with_interfaces + sched.overhead)

def get_oplist_time(op_list):
    return sum(op.time for op in op_list if not op.disabled)

class RK_Block_Solution:
    # A solution contains a fwd_sched and a bwd_sched
    # and also size_a_bar, time_fwd, time_bwd, overhead_fwd, overhead_bwd

    # prologue_op_list is added to the forward part of op_sched. Can be []
    # pred_disable is a predicate for ops that should be disabled from the schedules.
    #              Used to disable delete operations about the input data
    def __init__(self, block, sub_cluster, op_sched, prologue_op_list, pred_disable):
        
        def translate_and_filter(op_list):
            translated = translate(sub_cluster, op_list)
            return [ op for op in translated if not pred_disable(op) ]

        self.fwd_op_list = translate_and_filter(prologue_op_list + op_sched.op_list[: op_sched.loss_idx])
        self.bwd_op_list = translate_and_filter(op_sched.op_list[op_sched.loss_idx + 1 :])


        full_sched = OpSchedule(
            self.fwd_op_list
            + [ComputeOp(sub_cluster.loss_cnode)]
            + self.bwd_op_list,
            loss_idx=len(self.fwd_op_list),
            cluster=sub_cluster,
            correct_overhead=False,
        )

        # Include in abar size the size of output Tensors which are alive when computing loss
        self.size_a_bar = full_sched.mem + block.mem_output
        assert (block.mem_output == sum(anode.mem for anode in full_sched.interfaces["output_data_anodes"]
                                        if full_sched.alive_list[full_sched.loss_idx][anode.name]))

        self.time_fwd = full_sched.fwd_time
        self.time_bwd = full_sched.bwd_time
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

        # Since we have a sequential structure, we assume only the first compute node of the prologue has inputs
        input_anode_data = list(han.anode for han in h_compute_nodes[0].deps)
        output_anode_data = list(han.anode for han in h_compute_nodes[-1].users)
        self.mem_input = sum(anode.mem for anode in input_anode_data)
        self.mem_output = sum(anode.mem for anode in output_anode_data)

        prologue_op_list = []
        for f_hcn in previous_nodes:
            prologue_op_list += f_hcn.ff_op_list

        self.Fn_op_list = prologue_op_list + final_node.ff_op_list

        # Rotor assumes that input data is deleted only with Fn, so we remove delete operations from Fc 
        # and Fe (for Fe, we also prevent deletion of gradients of the input)
        def is_delete_of_input_data(op):
            return (isinstance(op, DeleteOp) and op.target in input_anode_data)

        def is_delete_of_input_data_or_gradient(op):
            return (isinstance(op, DeleteOp) and 
                    any(op.target.anode.main_target == anode.main_target for anode in input_anode_data))

        self.Fc_op_list = list(op for op in self.Fn_op_list if not is_delete_of_input_data(op))

        # One block solution for each FWD/BWD schedule
        self.sols = list(
            RK_Block_Solution(self, final_node.sub_cluster, op_sched, prologue_op_list, pred_disable=is_delete_of_input_data_or_gradient)
            for op_sched in get_sched(final_node.sub_cluster)
        )

        # TODO: measure both Fc and Fn peaks, they might be different
        self.overhead_fast_fwd = get_oplist_peak(self.Fc_op_list, final_node.sub_cluster) - self.mem_input - self.mem_output
        self.time_fast_fwd = get_oplist_time(self.Fc_op_list)

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
            # we only make a Block if the hcn has a BWD, otherwise we keep it to merge with the next one
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

        # The final output is not counted in the budget, so we should remove it from 
        #  the abar sizes.
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

        self.discretize()


    def discretize(self):
        def discretize_(values):
            return [int(math.ceil(value / self.mem_unit)) for value in values]

        self.cw = discretize_(self.cw)
        self.cbw = [discretize_(x) for x in self.cbw]
        self.fwd_tmp = [discretize_(x) for x in self.fwd_tmp]
        self.bwd_tmp = [discretize_(x) for x in self.bwd_tmp]
        self.ff_fwd_tmp = discretize_(self.ff_fwd_tmp)

    def _get_norecomputation_sequence(self):
        def cheapest_option(layer):
            return min(range(self.nb_sol[layer]), key=lambda i: self.fw[layer][i] + self.bw[layer][i])

        forward = []
        backward = []
        for i in range(self.ln):
            option = cheapest_option(i)
            forward.append(rkseq.SeqBlockFe(i, option, self.body[i].sols[option].fwd_op_list))
            backward.append(rkseq.SeqBlockBwd(i, self.body[i].sols[option].bwd_op_list))
        backward.reverse()
        return rkseq.RK_Sequence(forward + [rkseq.SeqLoss()] + backward )

    def _get_allrecomputation_sequence(self):
        def smallest_option(layer):
            return min(range(self.nb_sol[layer]), 
                       key=lambda i: self.cbw[layer+1][i]+max(self.fwd_tmp[layer][i], self.bwd_tmp[layer][i]))

        sequence = rkseq.RK_Sequence()

        for j in reversed(range(self.ln + 1)):
            sequence.insert(rkseq.SeqBlockFc(0, self.body[0].Fc_op_list))
            for i in range(1, j):
                sequence.insert(rkseq.SeqBlockFn(i, self.body[i].Fn_op_list))
            if j == self.ln:
                sequence.insert(rkseq.SeqLoss())
            else:
                option = smallest_option(j)
                sequence.insert(rkseq.SeqBlockFe(j, option, self.body[j].sols[option].fwd_op_list))
                sequence.insert(rkseq.SeqBlockBwd(j, self.body[j].sols[option].bwd_op_list))
        return sequence

    def get_min_memory(self):
        seq = self._get_allrecomputation_sequence()
        return self.simulate(seq, display=False) * self.mem_unit

    def get_max_memory(self):
        seq = self._get_norecomputation_sequence()
        return self.simulate(seq, display=False) * self.mem_unit

    def simulate(self, sequence, display=True, stopAtLoss=False):
        return sequence.simulate(self, display, stopAtLoss)

    def __str__(self):
        result = f"Chain of length {self.ln}\n"
        for i in range(self.ln):
            result += f"Block {i}: {self.cw[i]} {self.ff_fwd_tmp[i]} {self.ff_fw[i]:.2f} {self.nb_sol[i]}\n"
            for j in range(self.nb_sol[i]):
                peak_fw = self.cw[i] + self.cw[i+1] + self.fwd_tmp[i][j]
                peak_bw = self.cw[i] + self.cbw[i+1][j] + self.cw[i] + self.cw[i+1] + self.bwd_tmp[i][j]
                result += f"   Option {j}: {self.cbw[i+1][j]} {self.fwd_tmp[i][j]}:{peak_fw} {self.bwd_tmp[i][j]}:{peak_bw} {self.fw[i][j]:.2f} {self.bw[i][j]:.2f}\n"
        return result
