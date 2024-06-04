from exp import exp_rkmod, exp_pt, get3BPhi_2, exec_pt
from rockmate import Rockmate
import rockmate
import rkgb
import torch
import pickle
import exp


# from unet import UNet
# def getunet(batch_size, _, nlayers):
#     return (UNet(), [torch.randn(batch_size, 3, 224, 224)])

from GPT import GPT2
def getGPT2(batch_size, _, nlayers):
    # For GPT2-small
    d_model = 768
    n_head = 12
    return ( GPT2(nlayers=nlayers, d_model=d_model, n_head=n_head),
             [ torch.randint(50257, (batch_size, d_model)) ])

from torchvision.models import resnet50

def getresnet(batch_size, _, nlayers):
    return (resnet50(), [torch.randn(batch_size, 3, 224, 224)])

exp.Loss = lambda y: y.mean()

if __name__ == "__main__":
    
    if False:
        print("Testing")
        m, samples = getresnet(32, 10, 10)
        res = m(*samples)
        res.sum().backward()
        print("Done BWD")
        del res, m, samples

        print("With RK")
        exp_rkmod(nlayers=10, batch_size=32, id="resnet", exp_id="resnet_test",
                  activation_offload=False,
                  cpu_optimization=False, get_model=getresnet,
                  rotor=True, remat=True)
    else:
        nlayers = 10
        batch_size = 64
        get_model=getresnet
        filename_id = "resnet50"


        stats_pt = exp_pt(nlayers=nlayers,
                          batch_size=batch_size,
                          exp_id="7B_test",
                          get_model=get_model)

        device = torch.device("cuda")
        model, sample = get_model(batch_size, 512, nlayers=nlayers)
        model.to(device)
        sample = [s.to(device) for s in sample]

        ## budget = stats_pt["peak_mem"] - 1024*1024*1024
        budget = stats_pt["peak_mem"]
        print("Using budget:", budget)

        previous_result = None
        rkmod = None
        file_is_present = False

        partitioners = [rkgb.partitioned.PartitionerSequence(
        sub_partitioner=rkgb.partitioned.Partitioner())]
        solver = rockmate.solvers.HILP(ilp_solver="PULP_CBC_CMD")
        solver.config.offload = False
        solver.config.solve_only_top_level = False
        solver.config.nb_total_nodes_top_level = 0
        rk_solver = rockmate.solvers.RK_rotor()
        ## list_solvers = [solver, rk_solver]
        list_solvers = [rk_solver]

        try :
            print("Reading")
            with open(f"./{filename_id}_rkgb_res.pkl", "rb") as f:
                previous_result = pickle.load(f)
            print("Done reading")
            file_is_present = True
            rkmod = Rockmate(model, sample, budget, rkgb_res=previous_result, solve_sched=False,
                             ilp_solver="PULP_CBC_CMD",
                             list_solvers=list_solvers,
                             partitioners=partitioners,
                             ilp_time_limit=1*60,
                             minor_offload_size=10*1024**2,
            )
        except Exception as e:
            print("Did not find", e)
            
            rkmod = Rockmate(model, sample, budget, solve_sched=False,
                             ilp_solver="PULP_CBC_CMD",
                             list_solvers=list_solvers,
                             partitioners=partitioners,
                             ilp_time_limit=10*60,
                             minor_offload_size=10*1024**2,
            )
        finally:

            if False: ##Set to True for rendering
                rkmod.rkgb_res.hierarchical_structure.get_insights_of_the_structure()
                rkmod.rkgb_res.partitioned_structure.render(render_format="gv")
                for x in rkmod.rkgb_res.partitioned_structure.all_unique_clusters:
                    x.render(render_format="gv")

            cluster = rkmod.rkgb_res.hierarchical_cluster
            param_mem = sum(pnode.mem for pnode in cluster.parameter_nodes)
            param_grad_mem = sum(pnode.mem for pnode in cluster.parameter_nodes if pnode.info.requires_grad)
            act_budget = budget - param_mem - (1+rkmod.optimize_metrics["optimizer_states_size"]) * param_grad_mem
            if act_budget<0:
                print("Negative activation budget, unfeasible")
                exit(1)
            print("Solving with budget=", act_budget)

            rkmod.preprocess()
            rkmod.solve_sched(act_budget, recursive=not file_is_present)
            if not file_is_present:
                rkmod.save_to_local(".", filename_id)
            rkmod.get_compiled_fct()

        niters = 5
        stats_rotor = {}
        try:
            time, mem = exec_pt(rkmod, sample, niters=niters)
        except Exception as e:
            stats_rotor["exception"] = e
            raise e
        torch.cuda.synchronize()
        stats_rotor["time"] = time/niters
        stats_rotor["peak_mem"] = mem

        stats_rotor["theo_time"] = sum(op.time for op in rkmod.op_sched.op_list if isinstance(op, rockmate.op_schedule.ComputeOp))
        stats_pt["theo_time"] = sum(kcn.time for kcn in rkmod.rkgb_res.hierarchical_cluster.list_cnodes if kcn.time)
        
        print("alg", "theo_time", "theo_ratio", "time", "Tratio", "mem", "Mratio")
        print("PT ", stats_pt["theo_time"], 1, stats_pt["time"], 1, stats_pt["peak_mem"], 1)
        print("RK ", stats_rotor["theo_time"], stats_rotor["theo_time"]/stats_pt["theo_time"], stats_rotor["time"],  stats_rotor["time"]/stats_pt["time"],
              stats_rotor["peak_mem"], stats_rotor["peak_mem"]/stats_pt["peak_mem"])

