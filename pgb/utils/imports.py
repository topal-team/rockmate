# ====================
# = External imports =
# ====================

import ast
import astunparse
import torch
import numpy as np
from torch import tensor
import graphviz

# == rotor == for utils/def_inspection
import rotor.timing # -> use .make_timer
import rotor.memory # -> use .MeasureMemory
from rotor.memory import MemSize as rotor_MemSize
from rotor.inspection import tensorMsize as rotor_tensorMsize

# -> to support different versions of AST
import sys
svi = sys.version_info
py_version = svi.major + svi.minor/10

# ==========================
