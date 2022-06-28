import torch
import torch.nn as nn

from checkmate.core.dfgraph import DFGraph
from checkmate.core.graph_builder import GraphBuilder

try:
    from rotor.utils import *
except:
    from torch.hub import load_state_dict_from_url
    from rotor.utils import *
from rotor.timing import *
from rotor.memory import *


import read_trace_code
from read_trace_code import *
from tools_on_B_graph import *
from importlib import reload
import tools_on_B_graph as Btools

from toposort import toposort_flatten
import warnings
warnings.filterwarnings("ignore")

node_src = B_node(target="src",code="INPUT",is_input = True)
node_a = B_node(
    target="a",
    code="g(src)",
    required=[node_src])
node_b = B_node(
    target="b",
    code="f(a)",
    required=[node_a])
node_c = B_node(
    target="c",
    code="h(a,b)",
    required=[node_a,node_b])
var_c = B_var("c",node=node_c)

graph = B_graph()
# graph.nodes : Only for debugging : (tmp ?)
graph.nodes = [node_src,node_a,node_b,node_c]
# graph.outputs : You can get everything through this
graph.outputs = [var_c]

Dgraph = B_to_D(graph)
def g(x):
    return torch.clone(x)

def f(x):
    return torch.clone(x)

def h(x1,x2):
    return (x1+x2,x1-x2)

def bwd_node(node):
    bwd_code = ""
    if node.is_input:
        bwd_code += '{o}.backward({o}.grad)'.format(o=node.target)
    elif node.target_type==torch.Tensor:
        bwd_code += '{o}.backward({o}.grad, inputs={i})'.format(o=node.target, i='[%s]'%','.join([inp.target for inp in node.required_nodes]))
    elif node.target_type==tuple:
        bwd_code += 'for t in {o}:\n'.format(o=node.target)
        bwd_code += '    t.backward(t.grad,inputs={i})'.format(o=node.target, i='[%s]'%','.join([inp.target for inp in node.required_nodes]))
    else:
        raise AttributeError("Unknown edge type {t} {target}".format(t=type(exec(node.target)), target=node.target))
    setattr(node, 'bwd_code', bwd_code)

print([(node.target,node.is_input) for node in Dgraph.nodes])
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def forward():
    INPUT = torch.randn(10000,requires_grad=True,device='cuda')
    exec('src=INPUT')
    for node in sort_nodes:
        with torch.enable_grad(): 
            exec(node.code)
        setattr(node, 'target_type', type(locals()[node.target]))
        if node.target_type==torch.Size:
            setattr(node, 'is_Size')
        elif node.target_type==torch.Tensor:
            setattr(node, 'target_shape', locals()[node.target].shape)
            setattr(node, 'target_dtype', locals()[node.target].dtype)
        elif node.target_type==tuple:
            s = []
            d = []
            for t in locals()[node.target]:
                s.append(t.shape)
                d.append(t.dtype)
            setattr(node, 'target_shape', s)
            setattr(node, 'target_dtype', d)

forward()

for node in Dgraph.nodes:
    bwd_node(node)
print('sorted nodes:', [node.target for node in sort_nodes])

min_duration = 0

def generate_tensor(shape, type= torch.Tensor, dtype = torch.float32, device=torch.device('cuda')):
    if type==torch.Tensor:
        return torch.randint(high=1, size=shape, requires_grad=True, dtype=dtype,device=device)
    
    elif type==tuple:
        return tuple([torch.randint(high=1, size=s,requires_grad=True, dtype=d,device=device) for s,d in zip(shape,dtype)])

def inspection(node):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    timer = make_timer(device)
    memUsage = MeasureMemory(device)
    def perform_measure(func, prologue = None):
        def complete_func():
            if prologue: prologue()
            return func()
        _, usage, maxUsage = memUsage.measure(func)
        duration = timer.measure_median(func)
        if duration < min_duration:
            number_repetitions = 1 + int(min_duration // duration)
            duration = timer.measure_median(func, iterations = number_repetitions)
        return duration, int(usage), int(maxUsage)
    dir_local = {}
    def forwardOp():
        with torch.enable_grad():
            exec(node.code, globals(), dir_local)
    def backwardOp():
        exec(node.bwd_code, globals(), dir_local)
    # To create inputs
    for n in node.required_nodes:
        dir_local[n.target] = generate_tensor(n.target_shape, type= n.target_type, dtype = n.target_dtype)
    fwd_results = perform_measure(forwardOp)
    if node.target_type==torch.Tensor:
        dir_local[node.target].grad = generate_tensor(node.target_shape, type= node.target_type, dtype = node.target_dtype)
    elif node.target_type==tuple:
        for i,t in enumerate(dir_local[node.target]):
            t.grad = generate_tensor(node.target_shape[i], dtype = node.target_dtype[i])
    bwd_results = perform_measure(backwardOp)
    return fwd_results, bwd_results

# TODO: input is needed for backward while not in the required_nodes
INPUT = torch.randn(10000,requires_grad=True,device='cuda')
print([(node.target,inspection(node)) for node in Dgraph.nodes[::-1]])
