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
torch.autograd.set_detect_anomaly(True)
import rockmate
from rockmate import Rockmate
from rockmate.solvers import HILP, RK_rotor
import rkgb
import ast
import os
import sys
# sys.path.append(f'{os.environ["WORK"]}/rockmate-private-jg/')
sys.path.append(f'/home/ygusak/rockmate-private/')
from models import get_iterator_over_all_examples

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


gd = globals()
def forward_with_code(model, sample):
    dynamo_result = torch.export.export(model, args = tuple(sample))
    dynamo_signature = dynamo_result.graph_signature

    tmp_local = {}
    tmp_local["self"] = model

    for inp, param in dynamo_signature.inputs_to_parameters.items():
        tmp_local[inp] = model.get_parameter(param)

    for inp, param in dynamo_signature.inputs_to_buffers.items():
        tmp_local[inp] = model.get_buffer(param)

    assert(len(dynamo_signature.user_inputs)==len(sample))
    for d, s in zip(dynamo_signature.user_inputs, sample):
        tmp_local[d] = s

    for code in dynamo_result.graph.python_code("self").src.split("\n"):
        if not code:continue
        if "def forward" in code: continue
        if "return" in code: continue

        exec(code.strip(), gd, tmp_local)
    return tmp_local, [tmp_local[o] for o in dynamo_signature.user_outputs]

def test_dynamo_graph_execution_with_code(model, sample):
    try:
        tmp_local, outputs = forward_with_code(model, sample)
        logging.debug(f'Forward with code works')
        
        outputs[0].mean().backward()
        logging.debug(f'Backward after forward with code works')
    except Exception as e:
        logging.debug(f"Model based on TorchDynamo signature can't be trained: {e}") 
        raise RuntimeError

def test_dynamo_graph_execution(model, sample, **dynamo_kwargs):
    try:
        # model.eval()
        dynamo_result = torch.export.export(
                            model,
                            args = tuple(sample),
                            kwargs=None,
                            **dynamo_kwargs
                            )
        logging.debug('Dynamo graph builed with args=tuple(sample), kwargs=None  works')
        
        dynamo_module = dynamo_result.module()
        y = dynamo_module(*sample)
        logging.debug('Forward pass through Dynamo module works')

        loss = y.mean()
        loss.backward()
        logging.debug('Backward pass through Dynamo module works')

    except Exception as e:
        logging.debug(f"Propagation through Dynamo module doesn't work: {e}")
        raise RuntimeError

def test_dynamo_graph_builder(model, sample, **dynamo_kwargs):
    if True:
      try:
        dynamo_result : torch.export.ExportedProgram = torch.export.export(
                        model,
                        args = tuple(sample),
                        kwargs=None,
                        **dynamo_kwargs
                        )
        logging.debug('Dynamo graph builed with args=tuple(sample), kwargs=None  works')
      except Exception as e:
          logging.debug(f'Dynamo graph builder with args=tuple(sample), kwargs=None does not  work: {e}')
          raise RuntimeError
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
      raise RuntimeError

    # print(whole_code_ast)

def test_rkgb_graph_builder(*args, **kwargs):
    rkgb_res = rkgb.rkgb.Result(*args, **kwargs)
    return rkgb_res


if __name__=="__main__":
    test_remat = False

    if torch.cuda.is_available():
        torch.cuda.init()
        device = 'cuda'
    else:
        device = 'cpu'

    # device="cpu"
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
             "TFNO2d",
             ][::-1] # Fix MLP-mixer
    logging.info(f'Models to test: {examples}')

    iterator_over_all_examples = get_iterator_over_all_examples(device, examples=examples)
    #pdb.set_trace()

    while True:
        model, sample = None, [] # To do not accumulate memory
        logging.info(f'{"".join(["="]*60)}\n')
        try:
            name, model, sample, get_param_fct = next(iterator_over_all_examples)
            logging.debug(f"Model {name} has been built \n")
            

            # try:
            #     #Build graphs based on the partitioner
            #     test_dynamo_graph_builder(model, sample)
            #     logging.debug(f"== TorchDynamo graph for {name} has been built == \n")
            # except Exception as e:
            #     logging.debug(f"Torch Dynamo failed to build a graph!\n {e}")
                
            logging.info(f'{"".join(["-"]*10)} Test execution of TorchDynamo module for {name} model\n ')
            try:
                test_dynamo_graph_execution(model, sample)
                logging.debug(f"TorchDynamo module can be trained\n")
            except Exception as e:
                logging.debug(f"Torch Dynamo module can't be trained!\n {e}")

            
            logging.info(f'{"".join(["-"]*10)} Test execution with forward from TorchDynamo signature for {name} model\n ')
            try:
                test_dynamo_graph_execution_with_code(model, sample)
                logging.debug(f"Model based on TorchDynamo signature can be trained\n")
            except Exception as e:
                logging.debug(f"Model based on TorchDynamo signature can't be trained!\n {e}")

            logging.info(f'{"".join(["-"]*10)} Test RKGB for {name} model\n ')
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
                logging.debug(f"== RKGB graph for {name} has been built == \n")
                
                if test_remat:
                    try:
                        solver = HILP(ilp_solver="PULP_CBC_CMD")
                        solver.config.offload = False
                        rematMod = Rockmate(
                                model,
                                sample, 
                                budget=2e10, 
                                list_solvers=[solver])

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
                        #)
                    except Exception as e:
                        logging.debug(f'Remat has failed: {e}')

            except Exception as e:
                logging.debug(f"Graph builder problems! \n {e}")
                #extype, value, tb = sys.exc_info()
                #traceback.print_exc()
                #pdb.post_mortem(tb) 
                
                
        except Exception as e:
            logging.debug(e)
            logging.debug("=== End ===")
            break 

