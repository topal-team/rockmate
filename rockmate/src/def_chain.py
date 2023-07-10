# ==========================
# definition file of RK_Chain
# also contains RK_Chain builder -> depends on use_chk.py
#  based on rotor/algorithms/parameters.py
# ==========================
from rkgb.utils import imports_from_rotor as irotor
from rockmate.ILP_gurobi_solver import ModelGurobi
import numpy as np
from rockmate.def_op import RunOp, DelOp, OpSchedule
import math

# ==========================
# ======== RK Block ========
# ==========================


def get_rk_solution(list_kg, l_bd_abar, budget_all):

    if True: ##only support for Gurobi solver for now
        param_dict = {
            "LogToConsole": 0,
            "IntegralityFocus": 1,
        }
        md = ModelGurobi(
            list_kg[0],
            budget_all,
            max(l_bd_abar),
            gcd=10000,
            gurobi_params=param_dict,
        )

    else:
        md = ModelMIP(list_kg[0], budget_all, max(l_bd_abar), gcd=10000,)
        md.md.verbose = 0
    list_list_sols = []
    for bd_abar in np.sort(l_bd_abar)[::-1]:
        md.add_abar_constraint(bd_abar)
        md.solve()

        if not md.feasible:
            list_list_sols.append(False)
            continue
            # return False
        list_sols = []
        for kg in list_kg:
            fwd_sched, bwd_sched = md.schedule(kg)
            list_sols.append(RK_Block_Solution(fwd_sched, bwd_sched))
        list_list_sols.append(list_sols)
    return list_list_sols


class RK_Block_Solution:
    def __init__(self, fwd_sched, bwd_sched):
        self.fwd_sched, self.bwd_sched = fwd_sched, bwd_sched
        self.time_fwd = self.fwd_sched.time
        self.time_bwd = self.bwd_sched.time
        self.size_a_bar = self.fwd_sched.save[-1]
        self.overhead_fwd = self.fwd_sched.overhead
        self.overhead_bwd = (
            self.bwd_sched.overhead + self.bwd_sched.save[-1] - self.size_a_bar
        )


def get_rk_block(list_kg, nb_bdg_abar, nb_bdg_all):
    list_blocks = []
    for kg in list_kg:
        list_blocks.append(RK_Block(kg))
    kdn_sizes = [kdn.mem for kdn in kg.list_kdn]
    overheads = [kcn.overhead for kcn in kg.list_kcn]
    max_bdg = sum(kdn_sizes) + max(overheads)
    min_bdg = (
        list_blocks[-1].Fc_sched.overhead + list_blocks[-1].Fc_sched.save[-1]
    )
    l_bd_all = np.linspace(min_bdg, max_bdg, nb_bdg_all)
    sols = []
    uniq_sols = set()
    for bd_all in l_bd_all:
        l_bd_abar = np.linspace(kg.output_kdn_data.mem, bd_all, nb_bdg_abar)
        # for bd_abar in l_bd_abar:
        #     if bd_all >= bd_abar:
        list_sols = get_rk_solution(list_kg, l_bd_abar, bd_all)
        for sol in list_sols:
            if sol:
                t = (
                    sol[0].size_a_bar,
                    sol[0].overhead_fwd,
                    sol[0].overhead_bwd,
                )
                if not (t in uniq_sols):
                    uniq_sols.add(t)
                    sols.append(sol)
                    for s, block in zip(sol, list_blocks):
                        block.sols.append(s)
    return list_blocks


class RK_Block:
    def __init__(self, kg):
        self.block_name = (
            f"Block[{kg.input_kdn_data.name}->{kg.output_kdn_data.name}]"
        )
        self.sols = []
        # == build Fc/Fn schedule
        def _fast_fwd_sched():
            def _can_del(i, kdn):
                for kcn in kdn.users_real:
                    if "bwd" in kcn.name:
                        continue
                    if kg.list_kcn.index(kcn) > i:
                        return False
                return True

            op_list = []
            alive_list = []
            alive_status = np.zeros(len(kg.list_kdn) + 2, dtype=bool)
            alive_status[-1] = True
            loss_idx = kg.list_kcn.index(kg.loss_kcn)
            for i, kcn in enumerate(kg.list_kcn[:loss_idx]):
                op = RunOp(kcn)
                op.no_grad = True
                op_list.append(op)
                for kdn in kcn.users:
                    if "data" not in kdn.kdn_type:
                        continue
                    alive_status[kg.list_kdn.index(kdn)] = 1
                alive_list.append(alive_status.copy())
                for j, kdn in enumerate(kg.list_kdn):
                    if kdn in [kg.output_kdn_data, kg.output_kdn_grad]:
                        continue
                    if alive_status[j] and _can_del(i, kdn):
                        op = DelOp(kdn)
                        op.proxy = False
                        op_list.append(op)
                        alive_status[j] = 0
                        alive_list.append(alive_status.copy())
            return op_list, alive_list

        self.Fc_sched = OpSchedule(
            *_fast_fwd_sched(),
            kg.input_kdn_data,
            kg.input_kdn_grad,
            kg.output_kdn_data,
            kg.list_kdn,
            no_grad=True,
        )
        self.Fn_sched = OpSchedule(
            *_fast_fwd_sched(),
            kg.input_kdn_data,
            kg.input_kdn_grad,
            kg.output_kdn_data,
            kg.list_kdn,
            no_grad=True,
        )
        self.Fn_sched.get_del_input_idx(kg)
        self.Fn_sched.del_input()
        self.overhead_fast_fwd = self.Fc_sched.overhead
        self.time_fast_fwd = self.Fc_sched.time

        #  == build .mem_inp/out ==
        # memsize = lambda inp : kg.dict_info[inp].memsize
        self.mem_inp = kg.input_kdn_data.mem if kg.input_kdn_data.mem else 0
        self.mem_out = kg.output_kdn_data.mem if kg.output_kdn_data.mem else 0

    def __str__(self):
        return (
            f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
            f"{self.block_name} :\n"
            f"\tnb of sol : {len(self.sols)}\n"
            f"\tmem_inp   : {irotor.MemSize(self.mem_inp)}\n"
            f"\ttime_ff  : {self.time_ff}\n"
            f"\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n"
        )


# ==========================
# ======== RK CHAIN ========
# ==========================


class RK_Chain:
    def __init__(
        self,
        list_kg,
        eq_classes,
        nb_budget_abar=10,
        nb_budget_all=3,
        mem_unit=None,
    ):
        if mem_unit:
            self.mem_unit = mem_unit
        else:
            self.mem_unit = 1024 ** 2
        self.body = [None] * len(list_kg)
        for cls in eq_classes:
            l_kg = [list_kg[i] for i in cls]
            l_block = get_rk_block(l_kg, nb_budget_abar, nb_budget_all)
            for i, j in enumerate(cls):
                self.body[j] = l_block[i]

        # for l_kg in identical_kg:
        #     self.body += get_rk_block(l_kg, nb_budget_abar, nb_budget_all)

        # for g in list_kg:
        #     self.body.append(RK_Block(g,nb_budget_abar,nb_budget_all))
        #     print_debug(self.body[-1])
        # organizes the information for rotor_solver.py as in Rotor
        # -> fw/bw/cw/cbw/fwd_tmp/bwd_tmp
        # -> in those list, one dummy block is added at the end for Loss
        # fw/bw: runtime of fwd/bwd
        # cbw: saved memory in each solution
        # cw: saved memory for each checkpoint solution (only the input)

        # -- init variables --
        ln = len(self.body)
        mkl = lambda n: [[] for _ in range(n)]
        fw = mkl(ln + 1)
        bw = mkl(ln + 1)
        cw = [None] * (ln + 2)
        cbw = mkl(ln + 2)
        fwd_tmp = mkl(ln + 1)
        bwd_tmp = mkl(ln + 1)
        ff_fwd_tmp = [None] * (ln + 1)
        ff_fw = [None] * (ln + 1)
        nb_sol = []

        # -- extract info from each block
        for (i, b) in enumerate(self.body):
            nb_sol.append(len(b.sols))
            if nb_sol[-1] == 0:
                raise Exception(
                    f"We need at least one solution per block. "
                    f"Here {b.block_name} has no solution"
                )
            for sol in b.sols:
                fw[i].append(sol.time_fwd)
                bw[i].append(sol.time_bwd)
                cbw[i + 1].append(sol.size_a_bar)
                fwd_tmp[i].append(sol.overhead_fwd)
                bwd_tmp[i].append(sol.overhead_bwd)
            cw[i] = b.mem_inp
            ff_fwd_tmp[i] = b.overhead_fast_fwd
            ff_fw[i] = b.time_fast_fwd
        cw[ln] = self.body[-1].mem_out  #  the final output

        # for the Loss block :
        nb_sol.append(1)
        fw[-1] = [0]
        bw[-1] = [0]
        cw[-1] = 0
        cbw[-1] = [0]
        fwd_tmp[-1] = [0]
        bwd_tmp[-1] = [0]
        ff_fwd_tmp[-1] = 0
        ff_fw[-1] = 0

        #  return :
        self.ln = ln
        self.fw = fw
        self.bw = bw
        self.cw = self.discretize(cw)
        self.cbw = [self.discretize(x) for x in cbw]
        self.fwd_tmp = [self.discretize(x) for x in fwd_tmp]
        self.bwd_tmp = [self.discretize(x) for x in bwd_tmp]
        self.ff_fwd_tmp = self.discretize(ff_fwd_tmp)
        self.ff_fw = ff_fw
        self.nb_sol = nb_sol

    def discretize(self, values):
        return [math.ceil(value / self.mem_unit) for value in values]


# ==========================
