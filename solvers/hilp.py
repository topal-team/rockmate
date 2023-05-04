import time
import torch
import rkgb
from rkgb.utils import print_debug, np, irotor
from rkgb.utils.global_vars import ref_verbose, solver_name
from rkgb.utils.small_fcts import get_device
from rkgb.utils.ast_add_on import ast_to_str

from solvers.HILP_gurobi import ModelGurobi, solve_hg_recursive
from solvers.rotor_solver import seq_builder, solve_dp_functional
from solvers.op_schedule import OpSchedule, get_autograd_sched_rec


class HILP:
    def __init__(
        self,
        mem_unit=1024 ** 2,
        gurobi_params={"LogToConsole": 0, "IntegralityFocus": 1,},
    ):
        self.mem_unit = mem_unit
        self.gurobi_params = gurobi_params

    def solve(
        self,
        rkgb_res,
        mem_limit,
        recursive=True,
        print_info=False,
        protect_names=["sources data", "sources grad"],
        return_hg=False
    ):
        if isinstance(rkgb_res, rkgb.Htools.H_graph):
            return self.solve_hg(
                rkgb_res,
                mem_limit,
                mem_limit,
                print_info=print_info,
                protect_names=protect_names,
                
            )
        self.mem_limit = mem_limit
        #  -- build Hgraph --

        kg = rkgb_res.K_graph
        sg = rkgb_res.S_graph
        if recursive:
            ps = rkgb.Ptools.S_to_P(sg,None) # TO TODO None=model
            self.hg = rkgb.Htools.P_and_K_to_H(ps, kg)
            print(f"Size of Hgraph {len(self.hg.list_hcn)}")
            save_all_sched = get_autograd_sched_rec(self.hg, kg)
            self.hg.add_sched(save_all_sched)
            solve_hg_recursive(self.hg, solve_self=False, print_info=print_info)
            print("Low level finished")
        if return_hg:
            return self.hg
        self.md = ModelGurobi(
            self.hg,
            mem_limit,
            mem_limit,
            gurobi_params=self.gurobi_params,
            accurate_mem=True,
            protected_names=[
                kg.output_kdn_data.name
            ],  # output data is protected
        )
        self.md.solve()
        if not self.md.feasible:
            print("Not feasible solution")
            return OpSchedule([])
        else:
            print(f"Solution with obj: {self.md.md.getObjective().getValue()}")
        self.op_sched = self.md.schedule_()
        for op in self.op_sched.op_list:
            if op.name in protect_names:
                op.disabled = True
        return self.op_sched

    def solve_hg(
        self,
        hg: rkgb.Htools.H_graph,
        save_budget,
        peak_budget,
        print_info=False,
        protect_names=["sources data", "sources grad"],
        gurobi_params=None,
        accurate_mem=False,
    ):
        gurobi_params = gurobi_params or self.gurobi_params
        md = ModelGurobi(
            hg,
            save_budget,
            peak_budget,
            gurobi_params=gurobi_params,
            accurate_mem=accurate_mem,
        )
        md.solve()
        if md.feasible:
            op_sched = md.schedule_()
            for op in op_sched.op_list:
                if op.name in protect_names:
                    op.disabled = True
            if print_info:
                print(
                    f"Solve Hgraph {hg.name} with {len(hg.list_hcn)} nodes takes {md.solve_time:03f}s"
                )
            return op_sched
