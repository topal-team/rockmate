# ==========================
# This file is the root of the RK file 
# hierarchy. It contains the global vars
# and auxiliary functions. But also all
# the imports actions, even those specific
# ==========================


# ==========================
# ======== IMPORTS =========
# ==========================

# == pgb ==
import pgb
from pgb.utils import * # /!\

# == checkmate == -> for use_chk.py
from checkmate.core.graph_builder \
    import GraphBuilder         as CHK_GraphBuilder
from checkmate.core.schedule \
    import (ScheduledResult     as CHK_ScheduledResult, # -> useless
            ILPAuxData          as CHK_ILPAuxData,      # -> useless
            OperatorEvaluation  as CHK_OperatorEvaluation,
            DeallocateRegister  as CHK_DeallocateRegister,
            AllocateRegister    as CHK_AllocateRegister )
import cvxpy
from checkmate.core.solvers.cvxpy_solver \
    import solve_checkmate_cvxpy as CHK_solve_checkmate_cvxpy
from checkmate.plot.graph_plotting \
    import plot_schedule        as CHK_plot_schedule

try:
    from checkmate.core.solvers.gurobi_solver \
        import solve_ilp_gurobi as CHK_solve_ilp_gurobi
    gurobi_installed = True
except:
    gurobi_installed = False

# == annexe ==
import numpy as np
import matplotlib.pyplot as plt

# ==========================



# ==========================
# ====== GLOBAL VARS =======
# ==========================

ref_print_atoms = [True]

# ==========================



# ==========================
# ===== AUX FUNCTIONS ======

class piece_of_code():
    def __init__(self,str_code,name):
        self.str_code = str_code
        self.name = name
    def __repr__(self):
        return self.name
class bloc_of_code():
    def __init__(self,l,name):
        self.list_codes = l
        self.name = name
    def __repr__(self):
        s = "\n\t".join([c.__repr__() for c in self.list_codes])
        return f"{self.name}:\n\t{s}"
# ==========================
