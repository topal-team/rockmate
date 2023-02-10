# ==========================
# ====== H structure =======
# ==========================

from rkgb.utils import *

# The H_graph contains only forward part

# ************
# * H_C_node *
# ************

class H_C_node():
    def __init__(self,kg,number):
        self.kg     = kg
        self.number = number
        self.name   = f"Fwd {number}"
        self.deps   = set() # HDN set
        self.users  = set() # HDN set

# ************
# * H_D_node *
# ************

class H_D_node():
    def __init__(self,main_target):
        self.main_target = main_target
        self.name   = f"Data {main_target}"
        self.deps   = set() # HCN set
        self.users  = set() # HCN set

# ***********
# * K_graph *
# ***********

class K_graph():
    def __init__(self):
        self.dict_hn = dict()   # name -> HN
        self.list_hcn = []      # toposorted
        self.list_hdn = []
        self.last_hdn = None

    def make_users(self):
        for hn in self.dict_hn.values():
            for req_hn in hn.deps: req_hn.users.add(hn)
