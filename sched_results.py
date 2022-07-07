import torch
import torch.nn as nn

import read_trace_code
import Btools as Btools
from Btools import *
from D_gr_to_DF_gr import *
from importlib import reload

import numpy as np

import logging
from checkmate.core.graph_builder import GraphBuilder
from checkmate.core.schedule import ScheduledResult, ILPAuxData
from checkmate.core.solvers.cvxpy_solver import solve_checkmate_cvxpy
from checkmate.plot.graph_plotting import plot_schedule

def sched_by_checkmate(nodes,budget,plot_sched=False,solver='SCIPY'):
    gb = GraphBuilder()
    for k_node in nodes:
        # print(k_node.name)
        gb.add_node(name=k_node.name, cpu_cost=k_node.time, ram_cost=-k_node.fgt_mem.v)
    for k_node in nodes:
        for sub_n in k_node.req:
            if sub_n.fgt_mem:
                gb.add_deps(k_node.name,sub_n.name)
    g = gb.make_graph()
    print('total cost:', sum(g.cost_ram.values()), 'max cost:', max(g.cost_ram.values()))
    print('budget:', budget)

    sched_result = solve_checkmate_cvxpy(g, budget, solver_override='SCIPY', verbose =False)
    if sched_result.feasible: print('feasible schedule solved')

    if plot_sched: plot_schedule(sched_result)
    return sched_result


