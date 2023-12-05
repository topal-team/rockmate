import torch
import torch.nn as nn
from rockmate.solvers.op_schedule import *
from rockmate.solvers.main import add_parameter_node
from copy import deepcopy
from rkgb.utils import irotor
timer = irotor.make_timer(torch.device("cuda"))


def test_exec(model, sample, msg="", copy=False):
    torch.random.manual_seed(0)
    if msg:print(msg)
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad()
    mem = torch.cuda.memory_allocated()
    if copy:
        model = deepcopy(model).to("cuda")

    timer.start()
    for _ in range(10):
        if copy:
            y = model(*sample)
        else:
            y = model(*sample, record_mem = False)
        # print("output:", y.mean())
        loss = y.mean()
        loss.backward()
        model.zero_grad(set_to_none=True)

    timer.end()

    print(f"mean output {y.mean()}")
    # print(f"mean grad {model.get_parameter('0.weight').grad.mean()}")

    print(f"peak memory {(torch.cuda.max_memory_allocated() - mem)/1024**2:.0f}MB")
    print(f"time passed {timer.elapsed():.0f}ms")
    print()


def get_wide_decoder_NN(nlayers = 6, d_model = 4096, batch_size = 32, seq_length=40
    ):
    decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=32)
    model = nn.TransformerDecoder(decoder_layer, num_layers=nlayers).to("cuda")
    memory = torch.rand(seq_length, batch_size, d_model).to("cuda")
    tgt = torch.rand(seq_length, batch_size, d_model).to("cuda")
    sample = [tgt, memory]
    return model, sample

def prepare_for_offload(rkmod):
    rkmod.preprocess()
    print("finish preprocess")
    for hcn in rkmod.rkgb_res.H_cluster.possible_hg[0].list_hcn:
        add_parameter_node(hcn.sub_cluster, rkmod.original_mod)
    add_parameter_node(rkmod.rkgb_res.H_cluster, rkmod.original_mod)
    print("finish adding parameter node")


def disable_offload_op(rkmod):
    for op in rkmod.op_sched.op_list:
        if isinstance(op, OffloadOp):
            op.disabled = True
    rkmod.get_compiled_fct()

def disable_prefetch_op(rkmod):
    for op in rkmod.op_sched.op_list:
        if isinstance(op, PrefetchOp):
            op.disabled = True
    rkmod.get_compiled_fct()

def disable_delete_op(rkmod):
    for op in rkmod.op_sched.op_list:
        if isinstance(op, DeleteOp):# and "parameter" in op.target.name:
            op.disabled = True
    rkmod.get_compiled_fct()

def mod_to_cpu(model, grad=True, minor_size=10*1024):
    print("GPU memory before", torch.cuda.memory_allocated())
    for n,p in model.named_parameters():
        if p.numel()<minor_size:
            continue
        # if p.numel()>0:
        # print(n, p.data.shape, p.grad.shape)
        p.data = p.data.to("cpu")
        if grad:
            p.grad = None
        # print(torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    print("GPU memory after", torch.cuda.memory_allocated())

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
                if md.Comp[t, k, o].value() >0.9:
                    for w in range(W):
                        if md.AliveW[(t,k,w)].value()<1 or k in md.weight2hcn[w]:
                            print(f"\t{md.AliveW[(t,k,w)].value():.2%} of layer {w} is alive")


                    comp = f"{'fwd' if k<=T//2 else 'bwd'}-{min(k, T-1-k)}"
                    # print(f"Compute {md.hgraph.list_hcn[k]} with option {o} at stage {t}")
                    print(f"\tCompute {comp} with option {o} at stage {t}")

                    for w in range(W):
                        if md.OflW[(t,k,w)].value()>0:
                            print(f"\tOffload {md.OflW[(t,k,w)].value():.2%} of layer {w}")
                            # ofl_list.append(OflOp(md.hgraph.list_hdn[md.create_list[e][1]].kdn, 1, after=op_list[-1]))
                        
                        # src = md.create_list[e][0]
                        # if j<T-1:
                        #     if md.Alive[i,j+1,e].value() < md.PrfEnd[i,j,e].value()+(src==j)+(md.Alive[i,j,e].value()):
                        #         print(f"\tDelete {md.hgraph.list_hdn[md.create_list[e][1]]}")
                        #         # op_list.append(Op(md.hgraph.list_hdn[md.create_list[e][1]].kdn))
                    # for w in range(W):
                        if md.PrfW[(t,k,w)].value()>0:
                            print(f"\tPrefetch {md.PrfW[(t,k,w)].value():.2%} of layer {w}")
                                        
                        
                    # if md.PrfEnd[i,j,e].value()>0:
                    #     print(f"\tPrefetch done of edge {e}")