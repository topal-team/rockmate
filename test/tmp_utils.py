import numpy as np
import torch
import torch.nn as nn
from rockmate.solvers.op_schedule import (SynchronizeOp, 
                                          OffloadOp, 
                                          PrefetchOp,
                                          ComputeOp,
                                          DeleteOp,
                                          OptimizeOp)
# from rockmate.solvers.main import add_parameter_node
from copy import deepcopy
# from rkgb.utils import irotor
from rkgb.lowlevel.measure import TimerCUDA

from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

# timer = irotor.make_timer(torch.device("cuda"))
timer = TimerCUDA(torch.device("cuda"))

from models.GPT import  GPT2

class LoraLinear(nn.Module):
    def __init__(self, linear, num_adapters=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        u = nn.Parameter(torch.randn(num_adapters, self.linear.weight.shape[0]), requires_grad=True)
        v = nn.Parameter(torch.randn(self.linear.weight.shape[1], num_adapters), requires_grad=True)
        self.register_parameter("u", u)
        self.register_parameter("v", v)

    def forward(self, x):
        res1 = torch.matmul(x, self.v)
        res2 = torch.matmul(res1, self.u)
        y = self.linear(x)
        out = y+res2
        return out
    
# class LoraEmbedding(nn.Module):
#     def __init__(self, linear, num_adapters=10, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.linear = linear
#         self.linear.weight.requires_grad = False
#         self.linear.bias.requires_grad = False
#         u = nn.Parameter(torch.randn(self.linear.weight.shape[0], num_adapters).T)
#         v = nn.Parameter(torch.randn(num_adapters, self.linear.weight.shape[1]).T)
#         self.register_parameter("u", u)
#         self.register_parameter("v", v)

#     def forward(self, x):
#         res = torch.matmul(x, self.v)
#         res = torch.matmul(res, self.u)
#         return self.linear(x)+res

def manual_lora(model:nn.Module, target_modules, num_adapters=10, freeze_all=True):
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False
    for module_name in target_modules:
        module = model.get_submodule(module_name)
        if isinstance(module, nn.Linear):
            new_module = LoraLinear(module, num_adapters=num_adapters)
        # elif isinstance(module, nn.Embedding):
        #     new_module = LoraEmbedding(module, num_adapters=num_adapters)
        else:
            raise TypeError(f"manual lora does not work with {type(module)}")
        # setattr(model, module_name, new_module)

        atoms: List[str] = module_name.split(".")
        mod: torch.nn.Module = model

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                        "attribute `" + item + "`")
            if getattr(mod, item) == module:
                # setattr(mod, item, new_module)
                mod.add_module(item, new_module)
                break

            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                        "an nn.Module")




def find_op(rkmod, op_name, op_code=None, fct=None):
    for i, op in enumerate(rkmod.op_sched.op_list):
        if op_name in op.name or (
            hasattr(op,"kcn") and op_code and op_code in op.kcn.get_code()):
            print(i,op)
            if fct:
                fct(op)

def pop_ops(step):
    ops = []
    while step.opt_ops:
        ops.append(step.opt_ops.pop(0))
    return ops

def append_ops(step, ops):
    for op in ops:
        step.opt_ops.append(op)
    # return ops


def efficiency_test():
    a = np.random.randint(-1,2,[10000,8000])
    c = np.random.randint(0,10,[8000])
    d = 100
    x,y = 100,3000
    b = np.cumsum(a[:], axis=0)
    m = np.matmul(b, c)
    for _ in range(100):
        # b = np.cumsum(a[:], axis=0)
        
        s = a[x][d]
        a[x][d] = a[y][d]
        a[y][d] = s
        # b[:,d] = np.cumsum(a[:,d], axis=0)
        # m = np.matmul(b, c)

        m += (np.cumsum(a[:,d], axis=0) - b[:,d])*c[d]

def create_steps(rkmod):
    steps = []
    step = []
    for op in rkmod.op_sched.op_list:
        if isinstance(op, SynchronizeOp):
            steps.append(step)
            step = []
        step.append(op)
    steps.append(step)
        # print(op, op.time)
    step_time = []
    for s in steps:
        ofl_time = 0
        prf_time = 0
        opt_time = 0
        comp_time = 0
        for op in s:
            if isinstance(op, OffloadOp):
                ofl_time += op.time
            elif isinstance(op, PrefetchOp):
                prf_time += op.time
            elif isinstance(op, OptimizeOp) and "cpu" in op.name:
                opt_time += op.time
            elif isinstance(op, ComputeOp):
                comp_time += op.time
        step_time.append((ofl_time, prf_time, opt_time, comp_time))
    return steps, step_time

def save_mem(rkmod):
    param_size = 0
    act_size = 0
    view_params = []
    for k,v in rkmod.op_sched.dict_alloc_param.items():
        view_params += v.pnode.view_targets
    for k,v in rkmod.compiler.storage.ld.items():
        if not isinstance(v,torch.Tensor):continue
        if v.grad is not None and ("cuda" in str(v.device) or "cuda" in str(v.grad.device)):
            print(k)
        if "__" not in k and "cuda" in str(v.device):
            # if v.numel()>0:print(k,v.numel()*v.element_size())
            param_size += v.numel()*v.element_size()
            if v.grad is not None:
                print(k)
                param_size += v.grad.numel()*v.element_size()
        if "__" in k and k not in view_params:# and "___" not in k:
            if v.numel()>0:print(k,v.numel()*v.element_size())
            act_size += v.numel()*v.element_size()
            # if v.grad is not None:
            #     act_size += v.grad.numel()*v.element_size()

    optim_size_all = 0
    for k,optimizer in rkmod.compiler.storage.ld["optimizers"].items():#["minors"]
        optim_size = 0
        for k,v in optimizer.state.items():
            if not v:
                p_size= m_size=v_size=0
                continue
            else:
                if isinstance(k, torch.Tensor) and "cuda" in str(k.device):
                    p_size = k.numel()*k.element_size()
                # if isinstance(v["exp_avg"], torch.Tensor) and "cuda" in str(v["exp_avg"].device):
                    m_size = v["exp_avg"].numel()*v["exp_avg"].element_size()
                # if isinstance(v["exp_avg_sq"], torch.Tensor) and "cuda" in str(v["exp_avg_sq"].device):
                    v_size = v["exp_avg_sq"].numel()*v["exp_avg_sq"].element_size()
                else:
                    # print(k,v)
                    continue
            optim_size = m_size+v_size#+p_size#not with parameter itself
        if optim_size:
            # print(optim_size, p_size, m_size, v_size)
            optim_size_all += optim_size
    return param_size, act_size, optim_size_all

def measure_cost_large_model():
    niter = 5
    for n in [1,2]:
        model = GPT2(nlayers=n, vcb_sz=32000, d_model=4096, n_head=32)
        sample = [torch.randint(0, 5000, [8,256])]

        test_exec(model, sample, copy=True, niter=niter, opt=torch.optim.Adam, device="cuda")


def number_of_variables(md, exclude="Continuous"):
    type(md.md.variables()[0])
    str(md.md.variables()[0]).split("_")
    variables = dict()
    for var in md.md.variables():
        if var.cat == exclude:continue
        name = str(var).split("_")
        if name[0] not in variables:
            variables[name[0]] = ["_".join(name[1:])]
        else:
            variables[name[0]].append("_".join(name[1:]))
    for k, v in variables.items():
        print(k, len(v))

def optimize(rkmod, copy=False, lr=1e-6):
    opt = torch.optim.AdamW
    if copy:
        optimizer = opt(rkmod.parameters(), lr=lr)
        optimizer.step()
    else:
        prefetch_list = rkmod.compiler.compile_all_prefetch()
        for f in prefetch_list:
            f()
        torch.cuda.synchronize()
        # parameters = [rkmod.original_mod.get_parameter(kdn.main_target) for kdn in rkmod.rkgb_res.H_cluster.list_kdn_parameters]
        # optimizer = opt(params=parameters, lr=lr)
                
        # optimizer = opt(rkmod.parameters(), lr=lr)
        # optimizer.step()
        optimizers = []
        for hcn in rkmod.rkgb_res.H_cluster.possible_hg[0].list_hcn:
            if not hcn.is_fwd:continue
            sub_cluster = hcn.sub_cluster
            if sub_cluster:
                parameters = [rkmod.original_mod.get_parameter(kdn.main_target) for kdn in sub_cluster.list_kdn_parameters]
                optimizer = opt(params=parameters, lr=lr)
                optimizers.append(optimizer)
                # print(sub_cluster, len(parameters), len(optimizer.param_groups[0]), len(optimizers[-1].param_groups[0]["params"]))

            else:
                optimizers.append(None)
        parameters = []
        for n, p in rkmod.original_mod.named_parameters():
            if n not in [kdn.main_target for kdn in rkmod.rkgb_res.H_cluster.list_kdn_parameters]:
                parameters.append(p)
        optimizers.append(opt(parameters, lr=lr))
        # print(len(optimizers[0].param_groups[0]["params"]))
        for optimizer in optimizers:
            if optimizer:
                # print(len(optimizer.param_groups[0]["params"]))
                optimizer.step()
                # for p in optimizers[0].param_groups[0]['params']:
                #     p.data -= p.grad
    torch.cuda.synchronize()

def debug_mem_solution(rkmod):
    max_i, max_t, max_k = analyze_mem(rkmod)
    op_sched = rkmod.op_sched
    md = rkmod.list_solvers[0].md
    print(op_sched.op_name_list.index(f"({max_t}, {max_k})"))

    mem = 0
    for k, v in op_sched.alive_list[max_i].items():
        if v and "parameter" not in k:
            # print(k, op_sched.dict_alloc[k].mem)
            mem += op_sched.dict_alloc[k].mem
    print("parameter memory at {max_i}", mem)

def test_optimize_time(model, sample, device="cuda", opt=torch.optim.AdamW, opt_kwargs={"lr":1e-6}, niters=5):
    # batch = 1024*4
    # width = 1024*4
    # depth = 16
    # model = torch.nn.Sequential(*[nn.Linear(width, width)for _ in range(depth)]).to(device)
    # sample = [torch.ones([batch, width]).cuda()]
    model = model.to(device)
    sample = [s.to(device) for s in sample]
    size = 0
    for n,p in model.named_parameters():
        p.grad = torch.ones_like(p)
        size += p.numel()
    optimizer = opt(model.parameters(), **opt_kwargs)
    timer.start()

    for i in range(niters):
        optimizer.step()
    timer.end()
    print(f"{size*p.element_size()/1024**2:.1f}MB model")
    print(f"{niters} time optimization cost {timer.elapsed():.2f} ms")


def optimize_stats(parameter, opt=torch.optim.AdamW, opt_kwargs={"lr":1e-6}, niters=5):
    p = parameter.to("cuda")
    p.grad = parameter.to("cuda")
    size = p.numel()
    mem = torch.cuda.memory_allocated()
    optimizer = opt([p], **opt_kwargs)
    for i in range(niters):
        optimizer.step()
    mem_opt_states = torch.cuda.memory_allocated() - mem

    p = parameter.to("cpu")
    p.grad = parameter.to("cpu")
    optimizer = opt([p], **opt_kwargs)
    optimizer.step()

    timer.start()
    for i in range(niters):
        optimizer.step()
    timer.end()
    # print(f"{size*p.element_size()/1024**2:.1f}MB model")
    # print(f"{niters} time optimization cost {timer.elapsed():.2f} ms")
    return timer.elapsed()/niters/size, mem_opt_states

    
def plot_mem(mem_real, mem_theory, start=0, end=None):
    import matplotlib.pyplot as plt

    """
    for i,c,m in doc:
        mem_real.append(m)
        op = newmod.executor.op_list[i]
        if op.is_fgt:
            mem_theory.append(-op.n.del_mem.v)
        else:
            mem_theory.append(op.n.del_mem.v)
    """
    plt.plot(np.array(mem_real)[start:end], label="real")
    plt.plot(np.array(mem_theory)[start:end], label="theory")
    plt.legend()
    plt.show()


def analyze_mem_exec(i, rkmod, code=True, diff=False):
    print(f"allo memory: {rkmod.allo_mem[i]}")
    print(f"expt memory: {rkmod.expect_mem()[i] - rkmod.expect_mem()[i - 1]}")
    if diff:
        print(
            f"{rkmod.allo_mem[i]-(rkmod.expect_mem()[i] - rkmod.expect_mem()[i - 1])}"
        )
    if code:
        print(rkmod.full_code[i])


def alive_status_solution(rkmod, t, k):
    md = rkmod.list_solvers[0].md
    for w in range(md.W):
        print(f"layer {w} is {(md.AliveW[t,k,w]+md.PrfW[t,k,w]).value():.1%} alive")
    # for i in range(md.I):
    #     print(f"activation {i} is {(md.alive[t,k,i]).value():.1%} alive")


def analyze_mem(rkmod, print_status=False, with_param=True, with_grad=True, details=False):
    md = rkmod.list_solvers[0].md
    mem = {}
    for t in range(md.T):
        for k in md.krange(t):
            mem[t, k] = md.U[t, k].value()
            mem[t, k] += (
                sum(
                    md.Comp[t, k, o].value() * md.overhead[k][o]
                    for o in range(md.nR[k])
                )
                if md.sol(md.sumComp[t, k].value())
                else 0
            )
            mem[t, k] += sum(
                md.mem[i_] * md.delete[t, eidx_d].value() if hasattr(md.delete[t, eidx_d], "value") else md.delete[t, eidx_d]
                for eidx_d, (k_, i_) in enumerate(md.delete_list)
                if k == k_
            )
            act_multiplier = 1/((1-md.param_multiplier) 
                              if isinstance(md.param_multiplier, float) 
                              else (1-md.param_multiplier.value()))
            
            mem[t, k] *= act_multiplier
            if with_param:
                mem[t, k] += md.all_param_mem(t, k, with_multiplier=False).value()

            # for w in range(md.W):
            #     # mem[t,k] += 1*((md.AliveW[t,k,w]+md.PrfW[t,k,w]).value()>0)*md.parameter_size[w]
            #     mem[t,k] += (md.AliveW[t,k,w]+md.PrfW[t,k,w]).value()*md.parameter_size[w]

    max_t, max_k = max(mem, key=mem.get)
    max_i = np.argmax(rkmod.op_sched.save_mem + rkmod.op_sched.interface_mem + rkmod.op_sched.overhead)
    grad_size = 0#max(md.parameter_size)
    optimizer_states_mem = rkmod.op_sched.optimizer_states_size()*rkmod.gd['optimize_stats']['optimizer_states_size']
    optimizer_states_mem += sum([p.numel()*p.element_size() for p in rkmod.minor_parameters])*rkmod.gd['optimize_stats']['optimizer_states_size']
    print(
        f"solution peak memory {(max(mem.values())*md.gcd)/1024**2:.0f}MB at {max_t, max_k}"
    )
    print(
        f"op_sched peak memory {(rkmod.op_sched.get_peak_mem(with_interface=True, act_multiplier=act_multiplier) + optimizer_states_mem +with_grad*grad_size)/1024**2:.0f}MB"
    )
    return (max_i, max_t, max_k)

    if print_status:
        print("Solution status")
        alive_status_solution(rkmod, max_t, max_k)

        print("\nOp_sched status")
        for k, v in rkmod.op_sched.alive_list[max_i].items():
            print(k, v)


def test_exec(model_, 
              sample, 
              msg="", 
              copy=False, 
              record_mem=False, 
              niter=10, 
              opt=torch.optim.Adam, 
              opt_kwargs={"lr":1e-6},
              return_mod = False,
              device="cuda"):
    torch.random.manual_seed(0)
    if msg:
        print(msg)
    torch.cuda.reset_peak_memory_stats()
    sample = [s.to(device) for s in sample]
    mem = torch.cuda.memory_allocated()
    print(mem)
    if copy:
        model = deepcopy(model_).to(device)
        optimizer = opt(model.parameters(), **opt_kwargs)
    else:
        model = model_
    model.zero_grad()
    # init run
    if copy:
        model.zero_grad()
        y = model(*sample)
    else:
        y = model(*sample, record_mem=record_mem)
    loss = y.sum()
    loss.backward()
    if copy:
        optimizer.step()
        

    timer.start()
    for i in range(niter):
        model.zero_grad()
        if copy:
            y = model(*sample)
        else:
            y = model(*sample, record_mem=record_mem)
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize()
        print(f"output: {y.mean()}")
        # print(f"grad: {model.get_parameter('30.weight').grad.sum()}")
        # print(model)
        if copy:
            # optimize(model,copy=copy)
            optimizer.step()
        # opt = torch.optim.SGD
        # optimizer = opt(model.parameters(), lr=0.001)
        # optimizer.step()

        torch.cuda.synchronize()
        # print(f"bias: {model.get_parameter('30.bias')[0]}\n")
        # print(f"weight: {model.get_parameter('30.weight')[0,0]}\n")
        # del y
        # model.zero_grad(set_to_none=True)

    timer.end()

    # print(f"mean output {y.mean()}")
    # print(f"mean grad {model.get_parameter('0.parameter').grad.mean()}")

    print(f"peak memory {(torch.cuda.max_memory_allocated() - mem)/1024**2:.0f}MB")
    print(f"time passed {timer.elapsed():.0f}ms")
    if return_mod:
        return model

def get_wide_decoder_NN(nlayers=6, d_model=4096, batch_size=32, seq_length=40):
    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=32)
    model = nn.TransformerDecoder(decoder_layer, num_layers=nlayers).to("cuda")
    memory = torch.rand(seq_length, batch_size, d_model).to("cuda")
    tgt = torch.rand(seq_length, batch_size, d_model).to("cuda")
    sample = [tgt, memory]
    for n, p in model.named_parameters():
        p.grad = torch.empty_like(p)
    return model, sample


# def prepare_for_offload(rkmod):
#     rkmod.preprocess()
#     print("finish preprocess")
#     # for hcn in rkmod.rkgb_res.H_cluster.possible_hg[0].list_hcn:
#     #     add_parameter_node(hcn.sub_cluster, rkmod.original_mod)
#     add_parameter_node(rkmod.rkgb_res.H_cluster, rkmod.original_mod)
#     print("finish adding parameter node")


def disable_offload_op(rkmod):
    for op in rkmod.op_sched.op_list:
        if isinstance(op, OffloadOp) and op.grad:
            op.disabled = True
    rkmod.get_compiled_fct()


def disable_prefetch_op(rkmod):
    for op in rkmod.op_sched.op_list:
        if isinstance(op, PrefetchOp):
            op.disabled = True
    rkmod.get_compiled_fct()


def disable_delete_op(rkmod):
    for op in rkmod.op_sched.op_list:
        if isinstance(op, DeleteOp):  # and "parameter" in op.target.name:
            op.disabled = True
    rkmod.get_compiled_fct()


def mod_to_cpu(model, grad=True, minor_size=10 * 1024):
    # print("GPU memory before", torch.cuda.memory_allocated())
    for n, p in model.named_parameters():
        if p.numel() < minor_size:
            continue
        # if p.numel()>0:
        # print(n, p.data.shape, p.grad.shape)
        p.data = p.data.to("cpu")
        if grad:
            p.grad = None
        # print(torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    # print("GPU memory after", torch.cuda.memory_allocated())


def print_sched(rkmod):
    md = rkmod.list_solvers[0].md
    hg = md.hgraph

    op_list = []
    prf_list = []
    ofl_list = []

    T = md.T
    W = md.W

    for t in range(T):
        # if md.sumComp[(t,t)].value()!=1:
        #     print(f"Compute {t} is {md.sumComp[(t,t)].value()}")
        print(f"Stage {t}")

        for k in md.krange(t):
            for o in range(md.nR[k]):
                if md.Comp[t, k, o].value() > 0.9:
                    for w in range(W):
                        if md.AliveW[(t, k, w)].value() < 1 or k in md.param2hcn[w]:
                            print(
                                f"\t{md.AliveW[(t,k,w)].value():.2%} of layer {w} is alive"
                            )

                    comp = f"{'fwd' if k<=T//2 else 'bwd'}-{min(k, T-1-k)}"
                    # print(f"Compute {md.hgraph.list_hcn[k]} with option {o} at stage {t}")
                    print(f"\tCompute {comp} with option {o} at stage {t}")

                    for w in range(W):
                        if md.OflW[(t, k, w)].value() > 0:
                            print(
                                f"\tOffload {md.OflW[(t,k,w)].value():.2%} of layer {w}"
                            )
                        if md.OflG[(t, k, w)].value() > 0:
                            print(
                                f"\tOffload grad {md.OflW[(t,k,w)].value():.2%} of layer {w}"
                            )
                            # ofl_list.append(OflOp(md.hgraph.list_hdn[md.create_list[e][1]].kdn, 1, after=op_list[-1]))

                        # src = md.create_list[e][0]
                        # if j<T-1:
                        #     if md.Alive[i,j+1,e].value() < md.PrfEnd[i,j,e].value()+(src==j)+(md.Alive[i,j,e].value()):
                        #         print(f"\tDelete {md.hgraph.list_hdn[md.create_list[e][1]]}")
                        #         # op_list.append(Op(md.hgraph.list_hdn[md.create_list[e][1]].kdn))
                        # for w in range(W):
                        if md.PrfW[(t, k, w)].value() > 0:
                            print(
                                f"\tPrefetch {md.PrfW[(t,k,w)].value():.2%} of layer {w}"
                            )
                        if md.PrfG[(t, k, w)].value() > 0:
                            print(
                                f"\tPrefetch grad {md.PrfW[(t,k,w)].value():.2%} of layer {w}"
                            )

                    # if md.PrfEnd[i,j,e].value()>0:
                    #     print(f"\tPrefetch done of edge {e}")
