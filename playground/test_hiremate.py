'''
Script to test a remat strategy on one model

Re-materialization strategies: HiRemate, Rockmate

Models: Enc-Dec Transformer, Unet, FNO, UFNO, UNO

Training: single GPU, multi-GPU with data parallelism
'''

import pdb, traceback
import torch
import rockmate
from rockmate import Rockmate
from rockmate.solvers import HILP, RK_rotor
import rkgb
import ast
import os
import sys
sys.path.append(f'{os.environ["WORK"]}/rockmate-private-jg/')
from models import get_iterator_over_all_examples


def test_dynamo_graph_builder(model, sample, **dynamo_kwargs):
    if True:
        dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(sample),
                        kwargs=None,
                        **dynamo_kwargs
                        )
        print('Dynamo with args=tuple(sample), kwargs=None  works')
    if True:
        for input_key in ['x', 'src', 'input'][:]:
            try:
                input_dict = {input_key: sample[0]}
                dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(),
                        kwargs=input_dict,
                        **dynamo_kwargs
                        )
                print(f'Dynamo with args=tuple(), kwargs keys = {input_dict.keys()} works \n')
                break
            except:
                print(f'Key {input_key} does not work')
                continue

    dynamo_graph = dynamo_result.graph
    dynamo_signature = dynamo_result.graph_signature
    whole_code_str = dynamo_graph.python_code("self").src
    whole_code_ast : ast.FunctionDef = ast.parse(whole_code_str).body[0]

    # print(whole_code_ast)

def test_rkgb_graph_builder(*args, **kwargs):
    rkgb_res = rkgb.rkgb.Result(*args, **kwargs)
    return rkgb_res


if __name__=="__main__":
    
    if torch.cuda.is_available():
        torch.cuda.init()
        device = 'cuda'
    else:
        device = 'cpu'

    device="cpu"
    print(device)
    examples=[
             "GPT",
             "ResNet101",
             "nn_Transformer", #RKGB problems when initially put model on GPU (LSE not correctly alligned)
             "UNet",
             "RegNet32", #RKGB problems at "F" building step (invalid decimal literal)
             "FNO1d", #TorchDynamo and RKGB problem has been fixed after setting FNO1d to nn.Module (as nn.Sequential had problem with forwarding correct argument name through the sequence of blocks)
             #"FNO3d", #RKGB problems: during build_forward graph on code lines with 'slice'
             "UFNO", #RKGB problems: during build_forward graph on code lines with 'slice'
             "UNO", #TorchDynnamo & RKGB problems: with padding
            ][:] # Fix MLP-mixer
    print(examples)

    iterator_over_all_examples = get_iterator_over_all_examples(device, examples=examples)
    #pdb.set_trace()

    while True:
        model, sample = None, [] # To do not accumulate memory
        print(f'{"".join(["-"]*60)}')
        try:
            name, model, sample, get_param_fct = next(iterator_over_all_examples)
            print(f"== Model {name} has been built == \n")
            
            '''
            #solver = HILP(ilp_solver="PULP_CBC_CMD")
            #solver.config.offload = False
            #list_solvers = [solver]
            #budget =10**16
            #breakpoint()
            #rkmod = Rockmate(
            #    model,
            #    sample,
            #    budget=budget,
            #    list_solvers=list_solvers,
            #    rkgb_res=None,
            #    solve_sched=True,
                # verbose=False,
                # ilp_solver="PULP_CBC_CMD",
                # ilp_time_limit=1 * 60 // 360,
                # ilp_time_limit_top=10 * 60,
                # model_kwargs=None,
                # partitioners=partitioners,
                # max_size_S_graph_for_no_partitioning=40,
                # cpu_optim = torch.optim.Adam,
                # gpu_optim = torch.optim.Adam,
                # optim_kwargs = {},
                # minor_param_size = 10*1024**2,
    	    )

            #print("Success!!!")
            #continue
            '''

            try:
                #Build graphs based on the partitioner
                test_dynamo_graph_builder(model, sample)
                print(f"== TorchDynamo graph for {name} has been built == \n")
            except:
                print(f"Torch Dynamo problems!\n")
                
            try:
                # Why an error arise when model and sample are on 'cuda'
                rkgb_res = test_rkgb_graph_builder(
                                model,
                                model_args=sample,
                                # model_kwargs=model_kwargs,
                                # verbose=verbose,
                                # wanted_graphs={"FB"},
                                # partitioners=[partitioner],
                                inspection_device=torch.device("cuda"),
                                # print_time_in_each_stage=True
                            )
                # solver = HILP(ilp_solver="PULP_CBC_CMD")
                # solver.config.offload = False
                # rematMod = Rockmate(model, sample , budget=2e10, list_solvers=[solver])
                print(f"== RKGB graph for {name} has been built == \n")
            except:
                print(f"Graph builder problems! \n")
                #extype, value, tb = sys.exc_info()
                #traceback.print_exc()
                #pdb.post_mortem(tb) 
                
                
        except StopIteration:
            print("=== End ===")
            break 

