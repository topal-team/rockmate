import torch
import torch.nn as nn

import read_trace_code
from Dtools import *
from Stools import *
from Ktools import *
from importlib import reload

import numpy as np

import logging
from checkmate.core.graph_builder import GraphBuilder
from checkmate.core.schedule import ScheduledResult, ILPAuxData
from checkmate.core.solvers.cvxpy_solver import solve_checkmate_cvxpy
from checkmate.plot.graph_plotting import plot_schedule

def sched_by_checkmate(K_graph,budget,plot_sched=False,solver='SCIPY',verbose=False):
    gb = GraphBuilder()
    nodes = list(K_graph.dict_nodes.values())
    for k_node in nodes:
        gb.add_node(name=k_node.name, cpu_cost=k_node.time, ram_cost=-k_node.fgt_mem.v)
    # TODO: add constraint to not forget artifact nodes
    for k_node in nodes:
        for sub_n in k_node.req:
            if sub_n.fgt_mem:
                gb.add_deps(k_node.name,sub_n.name)
    g = gb.make_graph()
    print('total cost:', sum(g.cost_ram.values()), 'max cost:', max(g.cost_ram.values()))
    print('budget:', budget)

    sched_result = solve_checkmate_cvxpy(g, budget, solver_override=solver, verbose =verbose)
    if sched_result.feasible: print('feasible schedule solved')

    if plot_sched: plot_schedule(sched_result)
    return sched_result, g


