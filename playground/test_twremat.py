import rkgb, rockmate, torch
from torchvision.models import resnet18
from rockmate.solvers.twremat_utils import *

import twrematpy.twremat as twr
from twrematpy.utils import utils as twu
import networkx as nx


if __name__=="__main__":
    model, sample = resnet18(), torch.randn(2, 3, 224, 224)
    budget = 1024**9


    h_cluster = get_rockmate_graphs(model, sample)

    node_info, target, loss = get_twremat_graph(h_cluster) 
    steps_haskell = runtwremat(node_info, budget, target, loss)
   
    graph = twu.makeGraph(node_info)
    _, decomp = nx.algorithms.approximation.treewidth_min_degree(graph.to_undirected())

    sched = twr.remat(graph, decomp, target)
    filtered_sched = twr.filter_loss_recomputation(sched, loss, False)
    #steps_python = twu.greedy_optimization(budget, filtered_sched, node_info)
    steps_python = filtered_sched
    
    print(len(steps_haskell), len(steps_python))

    print(node_info)
    print(target, loss)