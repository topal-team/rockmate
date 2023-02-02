import torch
import rkgb

from rockmate.def_code import RunOp, DelOp

from checkmate.core.graph_builder import GraphBuilder as CHK_GraphBuilder
from checkmate.core.schedule import (
    ScheduledResult as CHK_ScheduledResult,
    ILPAuxData as CHK_ILPAuxData,
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

import numpy as np
import matplotlib.pyplot as plt

from models.GPT import get_GPT, GPT2

device = torch.device("cuda")
from experiments_on_checkmate.graph_builder import CHK_graph
import time
import pickle


def make_sched(
    kg: rkgb.Ktools.K_graph,
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

    for n in nodes:
        for sub_n in n.deps_real:
            chk_gb.add_deps(n.name, sub_n.name)
    chk_g = chk_gb.make_graph()

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
    pgb_res = rkgb.make_all_graphs(model, x)
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


import rockmate as rk
import numpy as np
import torch
from models.GPT import *

device = torch.device("cuda")

results = []
for n in np.arange(2, 12, 2):
    model = GPT2(nlayers=n, d_model=1600, n_head=25).to(device)
    x = torch.randint(0, 600, [2, 256]).to(device)
    max_budget = 150 + 100 * n
    newmod = rk.CheckpointedModule(
        model,
        x,
        max_budget,
        nb_budget_abar=20,
        nb_budget_all=20,
        get_sequence=False,
        get_code=False,
    )
    for budget in np.linspace(200, max_budget, 5) * 1024 ** 2:
        max_o = max(
            [kcn.overhead.v for kg in newmod.list_kg for kcn in kg.list_kcn]
        )
        newmod.get_sequence(max_o + budget)
        op_name_list = newmod.fwd_op_list + newmod.bwd_op_list
        results.append(
            [
                n,
                budget,
                newmod.ILP_solve_time + newmod.DP_solve_time,
                0,
                newmod.simulation_time,
                sum([kcn.time for kg in newmod.list_kg for kcn in kg.list_kcn]),
                [],
            ]
        )

with open("rockmate_exp_solve_time.pkl", "wb") as f:
    pickle.dump(results, f)


from rotor_exp.rotor import Checkpointable
import time
import numpy as np
from models.GPT import *

x = torch.randint(0, 600, [2, 256]).to(device)

results = []
for nlayers in np.arange(2, 4, 2):
    GPT2_0 = GPT2_input(d_model=model.d_model).to(device)
    GPT2_1 = GPT2_output(d_model=model.d_model)
    h = [
        TransformerBlock(d_model=model.d_model, n_head=model.n_head)
        for _ in range(nlayers)
    ]
    model_rotor = nn.Sequential(*h + [GPT2_1]).to(device)
    x_rotor = GPT2_0(x.clone().to(device)).requires_grad_()
    start = time.time()
    budget = 200 * 1024 ** 2
    chk = Checkpointable(model_rotor, verbosity=0)
    chk.measure(x_rotor)
    chk.compute_sequence(budget)
    end = time.time()

    results.append(
        [
            n,
            budget,
            end - start,
            0,
            chk.get_expected_makespan(),
            sum([kcn.time for kg in newmod.list_kg for kcn in kg.list_kcn]),
            [],
        ]
    )
with open("rotor_exp_solve_time.pkl", "wb") as f:
    pickle.dump(results, f)
