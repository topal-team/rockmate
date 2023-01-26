import torch
import pgb
from checkmate.core.graph_builder import GraphBuilder as CHK_GraphBuilder
from checkmate.core.schedule import (
    ScheduledResult as CHK_ScheduledResult,  # -> useless
    ILPAuxData as CHK_ILPAuxData,  # -> useless
    OperatorEvaluation as CHK_OperatorEvaluation,
    DeallocateRegister as CHK_DeallocateRegister,
    AllocateRegister as CHK_AllocateRegister,
)

from checkmate.plot.graph_plotting import plot_schedule as CHK_plot_schedule

try:
    from checkmate.core.solvers.gurobi_solver import (
        solve_ilp_gurobi as CHK_solve_ilp_gurobi,
    )

    gurobi_installed = True
except:
    gurobi_installed = False
import time
import pickle
import numpy as np

from experiments_on_checkmate.graph_builder import CHK_graph
from models.GPT import get_GPT, GPT2

device = torch.device("cuda")


def make_sched(
    kg: pgb.Ktools.K_graph,
    budget,
    plot_sched=False,
    solver="SCIPY",
    verbose=None,
    use_gurobi=True,
):
    if not gurobi_installed:
        use_gurobi = False

    chk_gb = CHK_GraphBuilder()
    nodes = kg.dict_nodes.values()
    for n in nodes:
        chk_gb.add_node(name=n.name, cpu_cost=n.time, ram_cost=n.run_mem.v)

    for n in nodes:  # -> build edges
        for sub_n in n.deps_real:
            chk_gb.add_deps(n.name, sub_n.name)
    chk_g = chk_gb.make_graph()

    # )
    if use_gurobi:
        sched_result = CHK_solve_ilp_gurobi(
            chk_g, budget, print_to_console=verbose, approx=False
        )
    op_list = []
    if sched_result.feasible:
        for op in sched_result.schedule:
            if isinstance(op, CHK_OperatorEvaluation):
                node_name = chk_g.node_names[op.id]
                node = kg.dict_nodes[node_name]
                # print(node_name, node.name)
                op_list.append([node, 1])

            elif isinstance(op, CHK_DeallocateRegister):
                node_name = chk_g.node_names[op.op_id]
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
    with open("checkmate_exp_GPT2.pkl", "wb") as f:
        pickle.dump(results, f)
