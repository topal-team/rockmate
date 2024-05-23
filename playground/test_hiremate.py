'''
Script to test a remat strategy on one model

Re-materialization strategies: HiRemate, Rockmate

Models: Enc-Dec Transformer, Unet, FNO, UFNO, UNO

Training: single GPU, multi-GPU with data parallelism
'''

import pdb
import torch
import rockmate
from rockmate import Rockmate
from rockmate.solvers import HILP, RK_rotor
import rkgb
import ast
import sys
sys.path.append('/home/ygusak/rockmate-private/')
from models import get_iterator_over_all_examples


def test_dynamo_graph_builder(model, sample, **dynamo_kwargs):
    try:
        dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(sample),
                        kwargs=None,
                        **dynamo_kwargs
                        )
        print('YOOOO/n')
    except:
        for input_key in ['x', 'src']:
            try:
                input_dict = {input_key: sample[0]}
                dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(),
                        kwargs=input_dict,
                        **dynamo_kwargs
                        )
                print('YHHHHH/n')
                break
            except:
                continue

    dynamo_graph = dynamo_result.graph
    dynamo_signature = dynamo_result.graph_signature
    whole_code_str = dynamo_graph.python_code("self").src
    whole_code_ast : ast.FunctionDef = ast.parse(whole_code_str).body[0]

    print(whole_code_ast)

def test_rkgb_graph_builder(*args, **kwargs):
    rkgb_res = rkgb.rkgb.Result(*args, **kwargs)
    return rkgb_res


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device='cpu'
    print(device)
    examples=[
            # "GPT",
            # "ResNet101",
            # "nn_Transformer",
            # "UNet",
            # "RegNet32", #RKGB problems at "F" building step
            "FNO1d", #RKGB problems: at "R" building step?
            # "FNO3d", #RKGB problems: should be nn.Module??
            # "UFNO", #RKGB problems
            # "UNO", #TorchDynnamo & RKGB problems
            ] # Fix MLP-mixer
    print(examples)

    iterator_over_all_examples = get_iterator_over_all_examples(device, examples=examples)
    pdb.set_trace()

    while True:
        model, sample = None, [] # To do not accumulate memory
        try:
            name, model, sample, get_param_fct = next(iterator_over_all_examples)
            print(f"== Model {name} has been built == \n")

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
                
        except StopIteration:
            print("=== End ===")
            break
