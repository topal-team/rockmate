from .utils import *

# k : index of a comp node
# i : index of a tensor
# j : index of a data node

class C_node():
    def __init__(self, kn):
        self.kn = kn
        self.required_data = None#req_real
        self.required_size = None#req_fake

class D_node():
    def __init__(self, kn, keep_kn=False, component: str,):
        if keep_kn: self.kn = kn
        assert component in ['data', 'grad', 'phantoms'], f"component cannot be {component}"
        self.name = ' '.join([kn.main_target, component])
        if component in ['data','grad']:
            self.mem = kn.fgt_mem.v
            self.tensor_targets = kn.tensor_targets
            self.info = kn.info
        if component == 'phantoms':
            self.mem = kn.run_mem.v - kn.fgt_mem.v

def K_to_CD(kn: K_node):
    cn = C_node(kn)
    dn0 = D_node(kn, component='data')
    cd_list = [cn, dn0]
    if kn.info.requires_grad:
        cd_list.append(D_node(kn, component='grad'))
    if kn.abar:
        cd_list.append(D_node(kn, component='phantoms'))
    return cd_list#TODO: put edges in the graph

class CD_graph():
    def __init__(self):
        # basic attributes :
        self.C_nodes    = [] #[k] : C_node
        self.D_nodes    = [] #[i] : D_node
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

