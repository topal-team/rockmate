# ==========================
# rockmate interface with checkmate
#  use checkmate to get the schedule
# and translate it from checkmate's R and S
# ==========================
import torch
import pgb

# from pgb.utils import * # /!\
from rockmate.def_code import RunOp, DelOp

#  == checkmate == -> for use_chk.py
from checkmate.core.graph_builder import GraphBuilder as CHK_GraphBuilder
from checkmate.core.schedule import (
    ScheduledResult as CHK_ScheduledResult,  # -> useless
    ILPAuxData as CHK_ILPAuxData,  # -> useless
    OperatorEvaluation as CHK_OperatorEvaluation,
    DeallocateRegister as CHK_DeallocateRegister,
    AllocateRegister as CHK_AllocateRegister,
)

# import cvxpy
# from checkmate.core.solvers.cvxpy_solver \
#     import solve_checkmate_cvxpy as CHK_solve_checkmate_cvxpy
from checkmate.plot.graph_plotting import plot_schedule as CHK_plot_schedule

try:
    from checkmate.core.solvers.gurobi_solver import (
        solve_ilp_gurobi as CHK_solve_ilp_gurobi,
    )

    gurobi_installed = True
except:
    gurobi_installed = False

#  == others ==
import numpy as np
import matplotlib.pyplot as plt

# from .def_code import Op, OpBlock
from models.GPT import get_GPT, GPT2

device = torch.device("cuda")
from experiments_on_checkmate.graph_builder import CHK_graph
import time
import pickle

# ==========================
# === make the schedule ====
# ==========================
# -> interface with checkmate


def make_sched(
    kg: pgb.Ktools.K_graph,
    budget,
    plot_sched=False,
    solver="SCIPY",
    verbose=None,
    use_gurobi=True,
):
    #  == verbose ==
    # if verbose is None: verbose = ref_verbose[0]
    # else: ref_verbose[0]=verbose

    # # == check solver ==
    # if solver not in cvxpy.installed_solvers():
    #     raise AttributeError(
    #         f"{solver} isn't installed ({cvxpy.installed_solvers()})")

    # == try to choose Gurobi ==
    if not gurobi_installed:
        use_gurobi = False

    # == build Checkmate graph ==
    chk_gb = CHK_GraphBuilder()
    nodes = kg.dict_nodes.values()
    for n in nodes:  # -> build nodes
        # if n.is_artefact:
        #     raise Exception("Artefact nodes not supported in CHK yet")
        # chk_gb.add_node(name=n.name, cpu_cost=1, ram_cost=n.run_mem.v)
        chk_gb.add_node(name=n.name, cpu_cost=n.time, ram_cost=n.run_mem.v)

    for n in nodes:  # -> build edges
        for sub_n in n.deps_real:
            chk_gb.add_deps(n.name, sub_n.name)
    chk_g = chk_gb.make_graph()

    #  == use Checkmate solver ==
    # print_debug(
    #     f"Ask checkmate to solve a graph with :"
    #     f"\n\ttotal cost : {sum(chk_g.cost_ram.values())},"
    #     f"\n\tmax   cost : {max(chk_g.cost_ram.values())},"
    #     f"\n\tbudget     : {budget}"
    # )
    if use_gurobi:
        sched_result = CHK_solve_ilp_gurobi(
            chk_g, budget, print_to_console=verbose, approx=False
        )
    # else:
    #     sched_result = CHK_solve_checkmate_cvxpy(
    #             chk_g, budget,
    #             solver_override=solver,
    #             verbose=verbose)

    # == result ==
    op_list = []
    if sched_result.feasible:
        # print_debug("*",end="")
        # print_debug('feasible schedule solved')
        for op in sched_result.schedule:
            if isinstance(op, CHK_OperatorEvaluation):
                node_name = chk_g.node_names[op.id]  # "fwd_"+main_target
                node = kg.dict_nodes[node_name]
                # print(node_name, node.name)
                op_list.append([node, 1])

            elif isinstance(op, CHK_DeallocateRegister):
                node_name = chk_g.node_names[op.op_id]  # "fwd_"+main_target
                node = kg.dict_nodes[node_name]
                op_list.append([node, 0])

            elif isinstance(op, CHK_AllocateRegister):
                pass

            else:
                raise TypeError(f"Operation not recognize:{type(op)}")

    if plot_sched:
        CHK_plot_schedule(sched_result)

    return sched_result, op_list, chk_g


def chk_exp(CHK_g, budget):

    start = time.time()
    sched_result, op_list, chk_g = make_sched(CHK_g, budget)
    if not sched_result.feasible:
        return (0, 0, 0, 0)
    op_name_list = []
    for op, run in op_list:
        if run:
            op_name_list.append(f"Run {op.name}")
        else:
            op_name_list.append(f"Del {op.name}")
    end = time.time()
    solve_time = sched_result.solve_time_s
    peak_mem = sched_result.schedule_aux_data.peak_ram
    train_time = sched_result.schedule_aux_data.cpu
    return (solve_time, peak_mem, train_time, op_name_list)


results = []
for n in np.arange(2, 12, 2):
    # 100 MiB output
    # 120 MiB per layer
    model = GPT2(nlayers=n, d_model=1600, n_head=25).to(device)
    x = torch.randint(0, 600, [2, 256]).to(device)
    max_budget = 150 + 100 * n
    pgb_res = pgb.make_all_graphs(model, x)
    kg = pgb_res.K_graph
    o_train_time = sum([kcn.time for kcn in kg.list_kcn])
    CHK_g = CHK_graph(kg)

    for budget in np.linspace(200, max_budget, 5) * 1024 ** 2:
        (solve_time, peak_mem, train_time, op_name_list) = chk_exp(
            CHK_g, budget
        )
        if peak_mem == 0:
            print("infeasile")
            results.append("infeasible")
        print(f"With {n} layers, budget {budget}, solved in {solve_time}")
        results.append(
            [
                n,
                budget,
                solve_time,
                peak_mem,
                train_time,
                o_train_time,
                op_name_list,
            ]
        )
    with open("checkmate_exp_noapprox.pkl", "wb") as f:
        pickle.dump(results, f)
