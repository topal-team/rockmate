from .utils import *
from .defs import RK_chain

def test_everything(nn_mod,dict_inputs,show_debug=False):
    # -- use pytorch graph builder to get the list of K_graphs --
    pgb_res = pgb.make_all_graphs(
        nn_mod,dict_inputs,
        show_debug=show_debug,
        bool_kg = False) # we don't need the whole K_graph
    list_kg = pgb_res.K_graph_list

    # -- use checkmate to solve all the blocks, and create the chain --
    chain = RK_chain(list_kg)
