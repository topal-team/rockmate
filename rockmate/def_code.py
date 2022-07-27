# ==========================
# contains definitions of raw code types
# contains RK_Functions_Builder
# -> which builds RK-equivalent of rotor/Checkpointable.functions
# ==========================

from .utils import *

# ==========================
# ======== RAW CODE ========
# ==========================
class RawAtomCode:
    def __init__(self,n : pgb.K_node,code)
        self.node = n
        self.code = code # -> temporary str, should be Ast

class RawBlockCode:
    def __init__(self,body):
        self.body = body # RawAtomCode list
# ==========================
