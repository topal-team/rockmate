def test_code(g : D_graph,nn_mod,dict_inputs : dict):
    loc_dict = {}
    loc_dict["self"] = nn_mod
    for inp in g.inputs:
        loc_dict[inp] = dict_inputs[inp]
    for v in g.dict_rand:
        exec(g.dict_rand[v], globals(), loc_dict)
    for n in g.nodes:
        if n.is_rand:
            for sub_t in n.req_rand:
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
        self.target_type = None # torch.Tensor or torch.size 
        self.target_size = None # depends on type
        self.requires_grad = None

class K_node():
    def __init__(self,name=None,code=None,req=None):
        self.name = name
        self.mem  = None
        self.time = None
        self.code = code
        self.req = req # list of K_node

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
    else:
        raise Exception(f"normally there should only be tensor or size at this point, but {tt} found")
    return info

def generate_val(info):
    tt = info.target_type
    if tt==torch.Size:
        return info.target_size
    else:
        assert(tt==torch.Tensor)
        return torch.randn(info.target_size,
            dtype=info.dtype,
            requires_grad=info.requires_grad,
            device=device)

def detach_code(n): # TODO TO IMPROVE
    code = (n.code).replace(n.target,"_"+n.target)
    return f"{code} ; {n.target} = _{n.target}.detach()"

def generate_bwd_code(n : D_node,info):
    tt = info.target_type
    assert(tt==torch.Size)
    if n.is_input:
        return '_{o}.backward({o}.grad)'.format(o=n.target)
    else:
        inputs_str = ','.join([inp.target for inp in n.req])
        return '_{o}.backward({o}.grad, inputs=[{i}])'.format(
                o=n.target,i=inputs_str)



def make_k_nodes(g : D_graph,nn_mod,dict_inputs : dict):
    # returns a list of K_nodes
    dict_info = {} # dict : D_node.target -> node_info
    dict_Kbw = {} # dict : D_node.target -> K_node(bw)
    dict_Kfw = {} # dict : D_node.target -> K_node(fw)

    # -- inputs --  
    for inp in g.inputs:
        dict_info[inp] = get_info(dict_inputs[inp])

    def handle_node(n : D_node):
        if n.is_input:
            pass # info already known
        else:
            # -- generate random inputs --
            inputs_Kbw = {}
            tmp_local = {"self" : nn_mod}
            for sub_n in n.req:
                sub_info = dict_info[sub_n.target]
                sub_x = generate_val(sub_info)
                tmp_local[sub_n.target] = sub_x
            if n.is_rand:
                for sub_r in n.req_rand:
                    exec(g.dict_rand[sub_r],globals(),tmp_local)
            # -- get info --
            exec(n.code , globals() , tmp_local)
            x = tmp_local[n.target]
            info = get_info(x)
            dict_info[n.target] = info

            # -- build Kfw --
            Kreq = [dict_Kfw[sub_n.target] for sub_n in n.req]
            code = n.code
            if info.requires_grad:
                code = detach_code(n)
            Kfw = K_node(name=n.target,code=code,req = Kreq)
            dict_Kfw[n.target] = Kfw

            # -- build Kbw --
            if info.requires_grad:
                assert(info.target_type == torch.Tensor)
                bwd_code = generate_bwd_code(n,info)
                Kbw = K_node(name=n.target, code=bwd_code, req=Kreq)
                dict_Kbw[n.target] = Kbw
                for sub_n in n.req:
                    if sub_n.target in dict_Kbw:
                        dict_Kbw[sub_n.target].req.append(Kbw)

            # -- inspection --
            fwd_mem, fwd_time, bwd_mem, bwd_time = inspection(n,info)
            






