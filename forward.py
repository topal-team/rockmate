def test_code(g : D_graph,nn_mod,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = nn_mod
    for inp in g.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for v in g.dict_rand:
        exec(g.dict_rand[v], globals(), loc_dict)
    for n in g.nodes:
        if n.is_rand:
            for sub_t in n.required_rand:
                exec(g.dict_rand[sub_t])
        if not n.is_input: exec(n.code, globals(), loc_dict)
    ret = []
    for out in g.outputs:
        ret.append(loc_dict[out])
    if len(ret)==1: return ret[0]
    else: return tuple(ret)


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Node_info():
    def __init__(self):
        self.dtype = None
        self.target_type = None # torch.Tensor, tuple, list or torch.size
        self.target_size = None # depends on type
        self.requires_grad = None
        self.sub_info = None

class K_node():
    def __init__(self):
        self.name = None
        self.mem  = None
        self.time = None
        self.code = None
        self.parents = None

def get_info(x) -> Node_info:
    info = Node_info()
    tt = type(x)
    info.target_type = tt
    if tt==torch.Size:
        info.target_size = x
        info.requires_grad = False
    elif tt==torch.Tensor:
        info.target_size = x.shape
        info.dtype = x.dtype
        info.requires_grad = x.requires_grad
    elif tt==list or tt==tuple:
        sub_l = [get_info(y) for y in x]
        info.sub_info = sub_l
        info.requires_grad = any([y.requires_grad for y in sub_l])
        # info.target_size = [y.target_size for y in sub_l]
        # info.dtype = [y.dtype for y in sub_l]
    else:
        raise Exception(f"sorry only tuple list tensor or size : {tt}")
    return info

def create_input(info):
    tt = info.target_type
    if tt==torch.Size:
        return info.target_size
    elif tt==torch.Tensor:
        return torch.randn(info.target_size,
                dtype=info.dtype,
                requires_grad=info.requires_grad,
                device=device)
    elif tt==list or tt==tuple:
        return tt([create_input(sub_i) for sub_i in info.sub_info])
    else:
        raise Exception(f"you should had the type {tt} here also")

def generate_bwd_code(n : D_node,info):
    tt = info.target_type
    assert(tt!=torch.Size)
    if tt==torch.Tensor:
        if node.is_input:
            return '{o}.backward({o}.grad)'.format(o=node.target)
        else:
            inputs_str = ','.join([
            return '{o}.backward({o}.grad, inputs={i})'.format(o=node.target,
                    i='[%s]'%','.join([inp.target for inp in node.required_nodes]))




def make_k_nodes(g : D_graph,nn_mod,dict_inputs : dict):
    # returns a list of K_nodes
    dict_info = {} # dict : D_node.target -> node_info

    # -- inputs --  
    for inp in g.inputs:
        dict_info[inp] = get_info(dict_inputs[inp])
"""
    # -- rand src nodes --
    for (tg,code) in g.dict_rand:
        tmp = {"self" : nn_mod}
        exec(code,globals(),tmp)
        x = tmp[tg]
        dict_info[tg] = get_info(x)
"""

    def handle_node(n : D_node):
        if n.is_input:
            pass # info already known
        else:
            # -- generate random inputs --
            tmp_local = {"self" : nn_mod}
            for sub_n in n.required_nodes:
                sub_info = dict_info[sub_n.target]
                sub_x = create_input(sub_info)
                tmp_local[sub_n.target] = sub_x
            if n.is_rand:
                for sub_r in n.required_rand:
                    exec(g.dict_rand[sub_r],globals(),tmp_local)
            # -- info --
            exec(n.code , globals() , tmp_local)
            x = tmp_local[n.target]
            info = get_info(x)
            dict_info[n.target] = info





