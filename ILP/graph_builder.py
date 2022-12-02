from .utils import *

# k : index of a comp node
# i : index of a tensor
# j : index of a data node

class CD_graph():
    def __init__(self):
        # basic attributes :
        self.k_nodes    = [] #[k] : K_node
        self.data_name  = [] #[i] : str f"{var name} {grad | data | phantoms}"
        self.time       = [] #[k]
        self.mem        = [] #[i]
        # relation attributes :
        self.CompInputDatas     = [] #[k] ~ req
        self.CompOutputDatas    = [] #[k] ~ out_d
        self.CompOutputTensors  = [] #[k] ~ out_t
        self.CompInputTensors   = [] #[k] ~ deps
        self.TensorOutputComps  = [] #[i] ~ Users
        self.DataToTensor       = [] #[j] ~ repr
        self.TensorToDatas      = [] #[i] ~ att
        self.DataInputComp      = [] #[j] ~ src

