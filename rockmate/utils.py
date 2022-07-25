# ==========================
# This file is the root of the RK file 
# hierarchy. It contains the global vars
# and auxiliary functions. But also all
# the imports actions, even those specific
# ==========================


# ==========================
# ======== IMPORTS =========
# ==========================

# quick start : same useful things as pytorch graph builder
from pytorch_gb.utils import *

# to use pytorch graph builder
import pytorch_gb as pgb

# == for use_chk.py == -> K_graph and checkmate
from pytorch_gb.Ktools import K_graph

# to solve checkmate
from checkmate.core.graph_builder import GraphBuilder
from checkmate.core.schedule import ScheduledResult, ILPAuxData
try:
    from checkmate.core.solvers.gurobi_solver import solve_ilp_gurobi
    gurobi_installed = True
except:
    gurobi_installed = False
from checkmate.core.solvers.cvxpy_solver import solve_checkmate_cvxpy
from checkmate.plot.graph_plotting import plot_schedule
import cvxpy

# to translate the schedule
from checkmate.core.schedule import (
    OperatorEvaluation,
    DeallocateRegister,
    AllocateRegister)

# ==

# ==========================



# ==========================
# ====== GLOBAL VARS =======
# ==========================


