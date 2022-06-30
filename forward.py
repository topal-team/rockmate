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

def flat(l):
    ret = []
    for sub in l:
        if isinstance(sub,list):
            flat_sub = flat(sub)
            ret.extend(flat_sub)
        else:
            ret.append(sub)
    return ret

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

def generate_val(info,inputs_Kbw,dict_Kbw_loc):
    # dict_Kbw_loc = dict_Kbw[sub_n.target]
    def aux(path,info):
        tt = info.target_type
        if tt==torch.Size:
            return info.target_size
        elif tt==torch.Tensor:
            random_x =  torch.randn(info.target_size,
                    dtype=info.dtype,
                    requires_grad=info.requires_grad,
                    device=device)
            inputs_Kbw[random_x] = dict_Kbw_loc[path]
            return random_x
        elif tt==list or tt==tuple:
            ret = []
            for i,sub_info in enumerate(info.sub_info):
                ret.append(aux(f"{i}_{path}",sub_info))
            return tt(ret)
        else:
            raise Exception(f"you should had the type {tt} here also")
    aux("",info)


def generate_bwd_code(n : D_node,info):
    tt = info.target_type
    assert(tt!=torch.Size)
    if tt==torch.Tensor:
        if node.is_input:
            return '{o}.backward({o}.grad)'.format(o=node.target)
        else:
            inputs_str = ','.join([
            return '{o}.backward({o}.grad, inputs={i})'.format(o=node.target,
                    i='[%s]'%','.join([inp.target for inp in node.req_nodes]))




def make_k_nodes(g : D_graph,nn_mod,dict_inputs : dict):
    # returns a list of K_nodes
    dict_info = {} # dict : D_node.target -> node_info
    dict_Kbw = {} # dict : D_node.target -> K_node(bw) dict
    dict_Kfw = {} # dict : D_node.target -> K_node(fw)

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
            inputs_Kbw = {}
            tmp_local = {"self" : nn_mod}
            for sub_n in n.req_nodes:
                sub_info = dict_info[sub_n.target]
                sub_x = generate_val(info,inputs_Kbw,dict_Kbw[sub_n.target])
                # -> added all sub_x's tensors in inputs_Kbw
                tmp_local[sub_n.target] = sub_x
            if n.is_rand:
                for sub_r in n.req_rand:
                    exec(g.dict_rand[sub_r],globals(),tmp_local)
            # -- info --
            # INSPECT FORWARD
            exec(n.code , globals() , tmp_local)
            x = tmp_local[n.target]
            info = get_info(x)
            dict_info[n.target] = info

            # -- build Kbw -- -> inputs_Kbw is very usefull
            # ---------------------------
            n_Kbw = {}
            x_flatten = flat_with_path(x)
            x_tensors = []
            for (key,tens) in x_flatten: # e.g. key = "0_1" or ""
                if not (isinstance(tens,torch.Size)):
                    assert(isinstance(tens,torch.Tensor))
                    if tens.requires_grad: x_tensors.append((key,tens))
            for (key,tens) in x_tensors:
                if tens in inputs_Kbw: # i.e. already exists 
                    n_Kbw[key] = inputs_Kbw[tens]
                else: # we need to create a new K_node(bw)


            """
            if n.fct=="list constructor":
                bw_nodes = {}
                for sub_n in n.req_nodes:
                    for i,sub_t in enumerate(x):
                        if tmp_local[sub_n.target] is sub_t:
                            for key,bwn in enumerate(dict_Kbw(sub_n.target)):
                                bw_nodes[f"{i}_{key}"] = bwn
                dict_Kbw[n.target] = bw_nodes
                # end
            elif n.fct=="getattr":
            """







