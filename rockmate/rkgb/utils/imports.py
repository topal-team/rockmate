# ====================
# = External imports =
# ====================

import ast
import astunparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from torch import tensor
import graphviz
import warnings
from copy import copy

# == rotor == for utils/def_inspection
import rkgb.utils.imports_from_rotor as irotor
#import rotor.timing # -> use .make_timer
#import rotor.memory # -> use .MeasureMemory
#from rotor.memory import MemSize as rotor_MemSize
#from rotor.inspection import tensorMsize as rotor_tensorMsize

# -> to support different versions of AST
import sys
svi = sys.version_info
py_version = svi.major + svi.minor/10

# ==========================
