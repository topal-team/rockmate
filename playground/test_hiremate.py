'''
Script to test a remat strategy on one model

Re-materialization strategies: HiRemate, Rockmate

Models: Enc-Dec Transformer, Unet, FNO, UFNO, UNO

Training: single GPU, multi-GPU with data parallelism
'''

import pdb
import traceback
import logging
import warnings
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

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def test_dynamo_graph_builder(model, sample, **dynamo_kwargs):
    if True:
      try:
        dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(sample),
                        kwargs=None,
                        **dynamo_kwargs
                        )
        logging.debug('Dynamo with args=tuple(sample), kwargs=None  works')
      except Exception as e:
          logging.debug(f'Dynamo with args=tuple(sample), kwargs=None does not  work: {e}')
    if False:
        for input_key in ['x', 'src', 'input'][:]:
            try:
                input_dict = {input_key: sample[0]}
                dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(),
                        kwargs=input_dict,
                        **dynamo_kwargs
                        )
                logging.debug(f'Dynamo with args=tuple(), kwargs keys = {input_dict.keys()} works \n')
                break
            except Exception as e:
                logging.debug(f'Key {input_key} does not work: {e}')
                continue

    try:
      dynamo_graph = dynamo_result.graph
      dynamo_signature = dynamo_result.graph_signature
      whole_code_str = dynamo_graph.python_code("self").src
      whole_code_ast : ast.FunctionDef = ast.parse(whole_code_str).body[0]
    except Exception as e:
      logging.debug(f'Dynamo export failed: {e}')

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
    logging.info(f'Device: {device}')
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
            ][::-1] # Fix MLP-mixer
    logging.info(f'Models to test: {examples}')

    iterator_over_all_examples = get_iterator_over_all_examples(device, examples=examples)
    #pdb.set_trace()

    while True:
        model, sample = None, [] # To do not accumulate memory
        logging.info(f'{"".join(["-"]*60)}')
        try:
            name, model, sample, get_param_fct = next(iterator_over_all_examples)
            logging.debug(f"== Model {name} has been built == \n")
            
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
                logging.debug(f"== TorchDynamo graph for {name} has been built == \n")
            except Exception as e:
                logging.debug(f"Torch Dynamo problems!\n {e}")
                
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
                logging.debug(f"== RKGB graph for {name} has been built == \n")
            except Exception as e:
                logging.debug(f"Graph builder problems! \n {e}")
                #extype, value, tb = sys.exc_info()
                #traceback.print_exc()
                #pdb.post_mortem(tb) 
                
                
        except Exception as e:
            logging.debug(e)
            logging.debug("=== End ===")
            break 

