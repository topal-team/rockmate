# ==========================
# contains definitions of raw code types
# contains RK_Functions_Builder
# -> which builds RK-equivalent of rotor/Checkpointable.functions
# ==========================

from .utils import *

# ==========================
# ======== RAW CODE ========
# ==========================
# -> output of use_chk.py
class RawAtomCode:
    def __init__(self,n : pgb.Ktools.K_node,code,is_fgt,is_fwd):
        self.node = n
        self.code = code # -> temporary str, should be Ast
        self.is_fgt = is_fgt
        self.is_fwd = is_fwd
        if is_fgt:  self.time = 0
        else:       self.time = n.time

class RawBlockCode:
    def __init__(self,body):
        self.body = body # RawAtomCode list
# ==========================
