# ==========================
# rockmate interface with checkmate
# use checkmate to get the schedule
# and translate it from checkmate's R and S
# ==========================

from .utils import *
from .def_code import Op, OpBlock
from ILP.graph_builder import CD_graph

# ==========================
# === make the schedule ====
# ==========================
# -> interface with checkmate

def make_sched(kg : pgb.Ktools.K_graph,budget,
        plot_sched=False,solver='SCIPY',
        verbose=None,use_gurobi=True,
        only_sched=True):
    # == verbose ==
    if verbose is None: verbose = ref_verbose[0]
    else: ref_verbose[0]=verbose

    cdgraph = graph_builder.build()#TODO
    solver = ModelGurobi(g=cdgraph, budget=budget, gcd=1e5)
    solver.solve()
    op_sched = solver.schedule()if solver.feasible else None
    if only_sched: return op_sched

def make_sched(kg : pgb.Ktools.K_graph,budget,
        plot_sched=False,solver='SCIPY',
        verbose=None,use_gurobi=True):
    # == verbose ==
    if verbose is None: verbose = ref_verbose[0]
    else: ref_verbose[0]=verbose

    # == check solver ==
    if solver not in cvxpy.installed_solvers():
        raise AttributeError(
            f"{solver} isn't installed ({cvxpy.installed_solvers()})")

    # == try to choose Gurobi ==
    if not gurobi_installed: use_gurobi=False

    # == build Checkmate graph ==
    chk_gb = CHK_GraphBuilder()
    nodes = kg.dict_nodes.values()
    for n in nodes: # -> build nodes
        if n.is_artefact:
            raise Exception("Artefact nodes not supported in CHK yet")
        chk_gb.add_node(name=n.name, cpu_cost=1, ram_cost=n.run_mem.v)
        #chk_gb.add_node(name=n.name, cpu_cost=n.time, ram_cost=n.run_mem.v)

    for n in nodes: # -> build edges
        for sub_n in n.req_real:
            chk_gb.add_deps(n.name,sub_n.name)
    chk_g = chk_gb.make_graph()

    # == use Checkmate solver ==
    print_debug(
        f"Ask checkmate to solve a graph with :"\
        f"\n\ttotal cost : {sum(chk_g.cost_ram.values())},"\
        f"\n\tmax   cost : {max(chk_g.cost_ram.values())},"\
        f"\n\tbudget     : {budget}")
    if use_gurobi:
        sched_result = CHK_solve_ilp_gurobi(
                chk_g, budget,
                print_to_console=verbose)
    else:
        sched_result = CHK_solve_checkmate_cvxpy(
                chk_g, budget,
                solver_override=solver,
                verbose=verbose)

    # == result ==
    op_list = [] 
    if sched_result.feasible:
        print_debug("*",end="")
        print_debug('feasible schedule solved')
        for op in sched_result.schedule:
            if isinstance(op, CHK_OperatorEvaluation):
                node_name = chk_g.node_names[op.id]#"fwd_"+main_target
                node = kg.dict_nodes[node_name]
                #print(node_name, node.name)
                op_list.append(Op(is_fgt=False,n=node))

            elif isinstance(op, CHK_DeallocateRegister):
                node_name = chk_g.node_names[op.op_id]#"fwd_"+main_target
                node = kg.dict_nodes[node_name]
                op_list.append(Op(is_fgt=True,n=node))

            elif isinstance(op, CHK_AllocateRegister):
                pass

            else:
                raise TypeError(f"Operation not recognize:{type(op)}")
    
    if plot_sched: CHK_plot_schedule(sched_result)
    
    return sched_result, op_list, chk_g

# ==========================
