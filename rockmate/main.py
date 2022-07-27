from .utils import *
from .def_chain import RK_Chain

def test_everything(nn_mod,dict_inputs,verbose=False):
    ref_verbose[0] = verbose
    # -- use pytorch graph builder to get the list of K_graphs --
    pgb_res = pgb.make_all_graphs(
        nn_mod,dict_inputs,
        verbose=verbose,
        bool_kg = False) # we don't need the whole K_graph
    list_kg = pgb_res.K_graph_list

    # -- use checkmate to solve all the blocks, and create the chain --
    rk_chain = RK_Chain(list_kg)
    rk_chain.build_rotor_chain()
