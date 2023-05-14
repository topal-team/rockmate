import torch
from rotor import timing
import time
import rockmate as rk
from rockmate.solvers import *
from rockmate.solvers.main import *
from rockmate.solvers.twremat import TwRemat
from rockmate.main import HRemat
from exp_utils import sanity_check

device = torch.device("cuda")
torch.random.manual_seed(0)

dtype = torch.float32

mod_tf = torch.nn.Transformer(
    nhead=16, num_encoder_layers=6, num_decoder_layers=6
).to(dtype)

mod_tf.to(device)

torch.random.manual_seed(0)

input_tf = [
    torch.rand((200, 64, 512), device=device).to(dtype),
    torch.rand((200, 64, 512), device=device).to(dtype),
]

rkMod = HRemat(
    mod_tf,
    input_tf,
    5000 * 1024**2,
    solve_sched=False,
    list_solvers=TwRemat(),
)


# for cluster in rkMod.rkgb_res.H_cluster.all_clusters:
#     if cluster is cluster.representee_cluster:
#         try:
#             TwRemat()(cluster, [10e9])
#         except Exception as e:
#             print(cluster.list_kcn)
#             print(cluster.list_kdn)
#             # raise e


def sanity_check(rkMod, inputs):
    rkMod.reinit()
    original_mod = deepcopy(rkMod.original_mod)
    inputs_o = deepcopy(inputs)
    inputs_r = deepcopy(inputs)
    torch.random.manual_seed(0)
    y_o = original_mod(*inputs_o)
    loss_o = y_o.sum()
    loss_o.backward()

    torch.random.manual_seed(0)
    y_r = rkMod(*inputs_r)
    loss_r = y_r.sum()
    loss_r.backward()

    if torch.allclose(loss_o, loss_r):
        print("Same loss obtained!")

    def compare_module_grad(module1, module2):
        same_grad = True
        for n, p in module1.named_parameters():
            # print(n)
            if not torch.allclose(
                module1.get_parameter(n), module2.get_parameter(n)
            ):
                print("Unequal weight found in:", n)
                same_grad = False

            if (
                module1.get_parameter(n).grad != None
                and module2.get_parameter(n).grad != None
            ):
                grad1 = module2.get_parameter(n).grad
                grad2 = module1.get_parameter(n).grad
                if not torch.allclose(grad1, grad2):
                    print("Unequal grad found in:", n)
                    print(torch.mean((grad1 - grad2) / grad1))
                    same_grad = False
        if same_grad:
            print("Same grad obtained!")

    compare_module_grad(rkMod.original_mod, original_mod)


def test(list_solvers):
    start = time.time()
    rkMod = HRemat(
        mod_tf,
        input_tf,
        5000 * 1024**2,
        solve_sched=True,
        list_solvers=list_solvers,
    )
    # print(rkMod.rkgb_res.H_cluster.list_kcn)
    # sanity_check(rkMod, input_tf)
    print(f"Solving time: {time.time()- start}")

    print(rkMod.op_sched.solver)
    print(rkMod.op_sched.simulation_overhead)

    rkMod.reinit()
    # rkMod.save_sched_to_local("")
    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()
    timer = timing.make_timer(device)
    timer.start()
    torch.random.manual_seed(0)
    for _ in range(10):
        y = rkMod.forward(*input_tf)
        loss = y.mean()
        loss.backward()
    timer.end()
    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f"Peak memory usage: {peak_mem}; given budget: {rkMod.budget}")
    print(f"Time: {timer.elapsed()}")
    # print(rkMod.original_mod.encoder.layers[0].linear1.weight.grad[0])


def test_original():
    timer = timing.make_timer(device)
    torch.random.manual_seed(0)

    torch.cuda.reset_peak_memory_stats()
    max_before = torch.cuda.max_memory_allocated()
    timer.start()
    torch.random.manual_seed(0)
    for _ in range(10):
        y = mod_tf.forward(*input_tf)
        loss = y.mean()
        loss.backward()
    timer.end()

    peak_mem = torch.cuda.max_memory_allocated() - max_before
    print(f"Peak memory usage: {peak_mem};")
    print(f"Time: {timer.elapsed()}")
    # print(mod_tf.encoder.layers[0].linear1.weight.grad[0])


# test_original()
test(list_solvers=[TwRemat(), HILP()])
# test(list_solvers=[TwRemat()])
# test(list_solvers=[HILP()])
