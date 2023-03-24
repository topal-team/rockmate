# ==========================
# ====== P structure =======
# ==========================

# TODO TODO ensure that a subgraph always requires_grad

# ** Graph partitioning **

from rkgb.utils import *
from rkgb.Stools import S_graph, S_node

def count_nb_subgraph(pg):
    nb = 0
    for pn in pg.list_nodes:
        if "Subgraph" in pn.name:
            nb += 1
    return nb

# **********
# * P_node *
# **********

class P_node():
    def __init__(self,
            main_graph,
            subgraph    = None,
            main_target = None,
            sn = None):
        self.main_graph  = main_graph
        self.subgraph    = sgp = subgraph
        self.main_target = mt  = main_target
        self.sn = sn # used for .deps/.user to compute io_targets
        assert((mt is not None) != (sgp is not None))
        if sgp is not None:
            self.subgraph_id = sid = sgp.graph_id
            self.name        = f"Subgraph_{sid}"
            self.is_leaf     = False
        else:
            self.subgraph_id = None
            self.name        = f"Var_{mt}"
            self.is_leaf     = True

        self.deps         = set()
        self.users        = set()
        self.deps_global  = set()
        self.users_global = set()
        # Global edges contain ALL the deps/users, of any depth
        self.unique_id = small_fcts.use_generator(
            main_graph.unique_id_generator,self)

    def size(self):
        if self.is_leaf: return 1
        else: return len(self.subgraph.list_nodes)
        
    def total_size(self):
        if self.is_leaf: return 1
        else: return self.subgraph.total_size()
        
    def deps_only_global(self):
        return self.deps_global - self.deps
    def users_only_global(self):
        return self.users_global - self.users

""" FOR FUTURE WORK, TO RECOGNIZE SIMILAR GRAPHS 
    def __eq__(self,pn2,force_order=False,raise_exception=False):
        pn1 = self
        b = small_fcts.check_attr(pn1,pn2,
            ["name","subgraph","main_target"],
            raise_exception)
        mmt = lambda nl : [pn.main_target for pn in nl]
        s = shared_methods.sort_nodes if force_order else (lambda s : s)
        for attr in ["deps","users",]:#"deps_global","users_global"]:
            b *= mmt(s(getattr(pn1,attr))) == mmt(s(getattr(pn2,attr)))
            if not b and raise_exception:
                raise Exception(f"P_nodes differ on attr {attr}")
        return bool(b)
    def __hash__(self):
        if hasattr(self,"unique_id"): return self.unique_id
        else: return id(self)
"""


# ***********
# * P_graph *
# ***********

class P_graph():
    def __init__(self,graph_id,unique_id_generator):
        self.graph_id = graph_id
        self.next_subgraph_id = 1 # for new sub_graphs' id
        self.unique_id_generator = unique_id_generator
        self.list_nodes = [] # contains only the nodes of self
        self.input_nodes  = set() # do NOT belong to self -> NOT in list_nodes
        self.output_nodes = set() # do belong to self -> in list_nodes

        # latter :
        self.input_targets = None
        self.output_targets = None
        self.all_produced = None
        self.all_used = None

    def size(self):
        return len(self.list_nodes)
    def total_size(self):
        tot = 0
        for pn in self.list_nodes:
            tot += pn.total_size()
        return tot

    def make_subgraph_id(self,new_graph_id):
        self.graph_id = new_graph_id
        next_id = 1
        for pn in self.list_nodes:
            if not pn.is_leaf:
                new_sg_id = f"{new_graph_id}_{next_id}"
                next_id  += 1
                pn.subgraph.make_subgraph_id(new_sg_id)
                pn.subgraph_id = new_sg_id
                pn.name        = f"Subgraph_{new_sg_id}"
        self.next_subgraph_id = next_id

    """ USELESS
    def make_dict_leaf_node_address(self):
        dict_address = self.dict_leaf_node_address
        for pn in self.list_nodes:
            if pn.is_leaf:
                dict_address[pn.main_target] = self.graph_id
            else:
                sub_pg = pn.subgraph
                sub_pg.make_dict_leaf_node_address()
                sub_dict = sub_pg.dict_leaf_node_address
                dict_address.update(sub_dict)
    """

    def is_last_wrapping_graph(self):
        return len(self.input_nodes) == 0
    
    def first_nodes(self): # -> users of at least one input
        spn = set(self.list_nodes)
        first_nodes = set()
        for inp_pn in self.input_nodes:
            first_nodes.update(spn.intersection(inp_pn.users_global))
        return first_nodes
    
    def make_all_used_and_all_produced(self):
        self.all_used = used = set()
        self.all_produced = produced = set()
        for pn in self.list_nodes:
            if pn.is_leaf:
                if not (pn.sn is None):
                    deps_sn = pn.sn.deps.keys()
                    deps_mt = [req_sn.main_target for req_sn in deps_sn
                               if not req_sn.is_artefact]
                    used.update(set(deps_mt))
                    produced.add(pn.main_target)
            else:
                sub_pg : P_graph = pn.subgraph
                sub_pg.make_all_used_and_all_produced()
                used.update(sub_pg.all_used)
                produced.update(sub_pg.all_produced)

    def make_input_targets(self):
        self.input_targets = self.all_used - self.all_produced
        for pn in self.list_nodes:
            if not pn.is_leaf:
                pn.subgraph.make_input_targets()

    def make_io_targets_attributes_of_subgraphs(self):
        self.make_all_used_and_all_produced()
        self.make_input_targets()
        # Explanation : 
        # -> inputs = used - produced
        # -> BUT outputs != produced - used. Since some tensors might
        # -> be used while still begin an output 
        # (A -> B -> C, both B and C can be outputs)
        all_interfaces = set() 
        # -> We gather all the inputs of the subgraphs
        # -> so we have all the interface nodes
        # -> so we can compute output_targets
        for pn in self.list_nodes:
            if not pn.is_leaf:
                all_interfaces.update(pn.subgraph.input_targets)
            else:
                all_interfaces.add(pn.main_target)
                if pn.sn is not None:
                    all_interfaces.update(set([
                        req_sn.main_target for req_sn in pn.sn.deps.keys()
                        if not req_sn.is_artefact
                    ]))
        for pn in self.list_nodes:
            if not pn.is_leaf:
                sub_pg : P_graph = pn.subgraph
                sub_pg.output_targets = (
                    sub_pg.all_produced.intersection(all_interfaces)
                )
                sub_pg.output_targets.update(sub_pg.all_produced - sub_pg.all_used)
                # -> produced tensors not used are automatically outputs
                # -> but normally we already took care of them the line above
                # -> except for the very last graph -> the output of the model
                # -> Not used by any one, but still an output
    

""" FOR FUTURE WORK, TO RECOGNIZE SIMILAR GRAPHS 
    def __eq__(self,pg2,force_order=False,raise_exception=False):
        pg1 = self
        b = small_fcts.check_attr(pg1,pg2,
            ["graph_id","dict_leaf_node_address"],
            raise_exception=raise_exception)
        mmt = lambda nl : [pn.main_target for pn in nl]
        s = shared_methods.sort_nodes if force_order else (lambda s : s)
        for attr in ["input_nodes","output_nodes"]:
            b *= mmt(s(getattr(pg1,attr))) == mmt(s(getattr(pg2,attr)))
            if not b and raise_exception:
                raise Exception(f"P_graphs differ on attr {attr}")
        b *= (len(pg1.list_nodes) == len(pg2.list_nodes))
        if not b and raise_exception:
            raise Exception("P_graphs differ on list_nodes length")
        if b:
            for pn1,pn2 in zip(pg1.list_nodes,pg2.list_nodes):
                b *= pn1.__eq__(pn2,force_order,raise_exception)
        return bool(b)
"""


# ************
# * P_config *
# ************

class merge_flow_option():
    def __init__(self,group):
        self.group = group
        self.make_nb_nodes_and_nb_subnodes()
        # -> self.nb_nodes
        # -> self.nb_subnodes
    def make_nb_nodes_and_nb_subnodes(self):
        self.nb_nodes    = len(self.group)
        self.nb_subnodes = sum(pn.size() for pn in self.group)


class P_config():
    def __init__(self,
            min_nodes_per_graph = 3,
            max_nodes_per_graph = 20,
            merge_flow_importance_nb_nodes = 1,
            merge_flow_importance_nb_subnodes = -1/4,
            merge_flow_constant_value = 5,
            merge_flow_value_fct = None,
            merge_flow_stop_condition = None,
            ):
        self.min_nodes_per_graph = min_nodes_per_graph  
        self.max_nodes_per_graph = max_nodes_per_graph

        # ** merge_flow **
        if merge_flow_value_fct is None:
            self.merge_flow_value_fct = lambda option : (
                math.exp(
                option.nb_nodes    * merge_flow_importance_nb_nodes 
              + option.nb_subnodes * merge_flow_importance_nb_subnodes
              + merge_flow_constant_value) )
        else:
            self.merge_flow_value_fct = merge_flow_value_fct
        if merge_flow_stop_condition is None:
            def merge_flow_stop_condition(pg,best_option):
                # True -> stop
                tot_nb_nodes = len(pg.list_nodes)
                if tot_nb_nodes <= max_nodes_per_graph: return True
                elif best_option.nb_subnodes > max_nodes_per_graph: return False
                else:
                    value = self.merge_flow_value_fct(best_option)
                    return (value <= (math.sqrt(tot_nb_nodes) * math.sqrt(self.max_nodes_per_graph)))
                    # -> no longer worth it to merge
                    # -> no idea if its a good stop condition
        self.merge_flow_stop_condition = merge_flow_stop_condition

default_config = P_config()

# =======================



# ==============================
# === WRAP, UNWRAP AND MERGE ===
# ==============================

# ********
# * WRAP *
# ********

# wrap a 'group' of nodes, currently living in 'main_pg'
# into a new node 'new_pn', adding one level of depth.
def wrap(group : list,main_pg : P_graph):
    group_nb = main_pg.next_subgraph_id
    main_pg.next_subgraph_id += 1
    new_pg = P_graph(
        graph_id=f"{main_pg.graph_id}_{group_nb}",
        unique_id_generator=main_pg.unique_id_generator
    )
    new_pg.list_nodes = group
    new_pn = P_node(
        main_graph = main_pg,
        subgraph   = new_pg,
    )
    set_group = set(group)
    all_input_nodes = set()
    new_pg.output_nodes = output_nodes = set()
    for pn in group:
        # ** link new_pn with global edges **
        new_pn.deps_global.update(pn.deps_global)
        new_pn.users_global.update(pn.users_global)
        # -> reciprocal at the end

        # -> change pn.main_graph
        pn.main_graph = new_pg

        # ** inputs ** 
        if not pn.deps.issubset(set_group):
            all_input_nodes.update(pn.deps)# at the end : minus set_group
            deps_outside = pn.deps - set_group
            for req_pn in deps_outside:
                req_pn.users.discard(pn)
                req_pn.users.add(new_pn)
                pn.deps.discard(req_pn)
                new_pn.deps.add(req_pn)
        
        # ** outputs **
        if (pn in main_pg.output_nodes
        or (not pn.users.issubset(set_group))):
            output_nodes.add(pn)
            user_outside = pn.users - set_group
            for user_pn in user_outside:
                user_pn.deps.discard(pn)
                user_pn.deps.add(new_pn)
                pn.users.discard(user_pn)
                new_pn.users.add(user_pn)

    # ** reciprocal global edges **
    for req_g_pn in new_pn.deps_global:
        req_g_pn.users_global.add(new_pn)
    for user_g_pn in new_pn.users_global:
        user_g_pn.deps_global.add(new_pn)
    
    # ** input_nodes **
    new_pg.input_nodes = all_input_nodes - set_group

    # ** update main_pg.list_nodes **
    # I ASSUME that new_pn can take the place
    # of any former node, I'm 95% confident
    main_lpn = main_pg.list_nodes
    main_lpn[main_lpn.index(group[0])] = new_pn
    for pn in group[1:]:
        main_lpn.remove(pn)

    # ** update main_pg outputs **
    main_out = main_pg.output_nodes
    if not main_out.isdisjoint(set_group):
        main_out.add(new_pn)
    main_pg.output_nodes -= set_group
    return new_pn


# **********
# * UNWRAP *
# **********

# unwrap 'pn' in its main graph
def unwrap(pn : P_node):
    pg      : P_graph = pn.subgraph
    main_pg : P_graph = pn.main_graph
    if pn.is_leaf: return ()
    if main_pg.is_last_wrapping_graph(): return ()
    group = list(pg.list_nodes)

    # ** unplug pn/pg **
    # -> direct edges
    copy_pn_deps  = list(pn.deps)
    copy_pn_users = list(pn.users)
    for req_pn in pn.deps: req_pn.users.remove(pn) # useless
    for user_pn in pn.users: user_pn.deps.remove(pn) # useless
    # redondant with the update of deps and users 12 lines below
    # -> global edges
    for req_g_pn in pn.deps_global: req_g_pn.users_global.remove(pn)
    for user_g_pn in pn.users_global: user_g_pn.deps_global.remove(pn)

    # ** plug back the group **
    # -> fix main_pg.list_nodes
    main_lpn = main_pg.list_nodes
    i = main_lpn.index(pn)
    main_pg.list_nodes = main_lpn[:i] + group + main_lpn[i+1:]    
    # -> fix sub_pn.main_graph
    for sub_pn in group:
        sub_pn.main_graph = main_pg
    # -> use the property : deps = deps_global inter list_nodes 
    main_spn = set(main_pg.list_nodes)
    to_update = group + copy_pn_deps + copy_pn_users
    for sub_pn in to_update:
        sub_pn.deps  = sub_pn.deps_global.intersection(main_spn)
        sub_pn.users = sub_pn.users_global.intersection(main_spn)
        if sub_pn.users != sub_pn.users_global: # could just check the size
            main_pg.output_nodes.add(sub_pn)

    # ** update main_pg inputs/outputs **
    """ No need to change main_pg.input_nodes 
    main_pg.input_nodes = (
        set(main_pg.input_nodes).union(
        set(pg.input_nodes) - main_spn
        )) """
    if pn in main_pg.output_nodes:
        main_pg.output_nodes.remove(pn)
    # -> We already added new outputs 8 lines above
    return ()


# *********
# * MERGE *
# *********

# Merging replaces a group of nodes living in 'main_pg'
# by one new node 'new_pn', with :
# .list_nodes = \sum sub_pn.list_nodes for sub_pn in group
# thus it creates one wrapper, but flats the first depth level
# -> To do so, wrap and then unwrap each sub_node
def merge(group : list, main_pg : P_graph):
    # print("nb before wrap:",count_nb_subgraph(main_pg))
    new_pn = wrap(group,main_pg)
    # print("nb after wrap:",count_nb_subgraph(main_pg))
    new_pg = new_pn.subgraph
    # print("main_graph : ",main_pg)
    # print("new pg : ",new_pg)
    original_lpn = new_pg.list_nodes
    for sub_pn in original_lpn:
        #print(sub_pn.main_graph)
        unwrap(sub_pn)
    # print("nb after unwrap:",count_nb_subgraph(main_pg))
    main_pg.make_subgraph_id(main_pg.graph_id)
    return new_pn

# ==========================



# =============================================
# === Generate a P_graph based on a S_graph ===
# =============================================

def S_to_P_init(sg):
    unique_id_generator = [0]
    pg = P_graph("0",unique_id_generator)
    pg.list_nodes = lpn = []
    dict_mt_to_pn = dict()
    for sn in sg.nodes:
        if sn.is_artefact: continue
        mt = sn.main_target
        pn = P_node(
            main_graph  = pg,
            main_target = mt,
            sn = sn)
        lpn.append(pn)
        dict_mt_to_pn[mt] = pn
        for req_sn in sn.deps.keys():
            if not (req_sn is sg.init_node or req_sn.is_artefact):
                req_pn = dict_mt_to_pn[req_sn.main_target]
                pn.deps.add(req_pn)
                req_pn.users.add(pn)
    for pn in lpn:
        pn.deps_global = set(pn.deps)
        pn.users_global = set(pn.users)

    # ** input ** 
    last_wrapping_graph = P_graph("-1",unique_id_generator)
    input_pn = P_node(
        last_wrapping_graph,
        main_target="sources"
    )
    last_wrapping_graph.list_nodes = [input_pn]
    pg.input_nodes = set([input_pn])
    for user_sn in sg.init_node.users:
        user_pn = dict_mt_to_pn[user_sn.main_target]
        user_pn.deps_global.add(input_pn)
        input_pn.users_global.add(user_pn)
    
    # ** output **
    output_pn = dict_mt_to_pn[sg.output_node.main_target]
    pg.output_nodes = set([output_pn])
    return pg

# ==========================



# =====================================================
# === FIRST RULE : GROUP SEQUENCE OF NODES TOGETHER ===
# =====================================================

def rule_group_sequences(pg : P_graph,config : P_config = default_config):
    # ** Find the sequences **
    tot_nb_seq = 0
    dict_seq_nb = dict() # mt -> a seq nb
    dict_sequences = dict() # seq nb -> list of nodes in the seq
    for pn in pg.list_nodes:
        if len(pn.users) == 1 and len(list(pn.users)[0].deps) == 1:
            mt = pn.main_target
            user_pn = list(pn.users)[0]
            user_mt = user_pn.main_target
            if mt in dict_seq_nb:
                seq_nb = dict_seq_nb[mt]
                dict_seq_nb[user_mt] = seq_nb
                dict_sequences[seq_nb].append(user_pn)
            else:
                tot_nb_seq += 1
                dict_seq_nb[mt] = tot_nb_seq
                dict_seq_nb[user_mt] = tot_nb_seq
                dict_sequences[tot_nb_seq] = [pn,user_pn]

    # ** split too long sequences **
    all_sequences = list(dict_sequences.items())
    for seq_nb,sequence in all_sequences:
        seq_len = len(sequence)
        if seq_len > config.max_nodes_per_graph:
            del dict_sequences[seq_nb]
            nb_seq = math.ceil(seq_len/config.max_nodes_per_graph)
            sub_seq_len = math.ceil(seq_len/nb_seq)
            for first in range(0,seq_len,sub_seq_len):
                end = min(first + sub_seq_len,seq_len)
                sub_seq = sequence[first:end]
                tot_nb_seq += 1
                sub_seq_nb = tot_nb_seq
                dict_sequences[sub_seq_nb] = sub_seq
                for pn in sub_seq:
                    dict_seq_nb[pn.main_target] = sub_seq_nb

    # ** Group each sequence **
    for seq_nb,sequence in dict_sequences.items():
        if len(sequence) >= config.min_nodes_per_graph:
            new_pn = wrap(sequence,pg)

    pg.make_subgraph_id(pg.graph_id)

# ==========================



# ===============================================================
# === SECOND RULE : MERGE NODES WITH A UNIQUE COMMON ANCESTOR ===
# ===============================================================

# the flow of pn is the list of nodes 
# to_be_visited which are descendants of pn

def rule_merge_small_flows(pg : P_graph,config : P_config = default_config):
    # === FIRST ===
    # for each node we need to find where its flow converge back
    dict_nb_usages = dict([(pn, len(pn.users)) for pn in pg.list_nodes])
    to_be_visited = []
    dict_flow = dict()
    # for a pn already visited -> its descendants in to_be_visited
    # if len(flow_size) = 0 => the flow converged
    # its a generalisation of "seen" in cut_based_on_deps
    dict_total_flow = dict()
    # when a node is popped out of to_be_visited, 
    # its removed from dict_flow but dict_total_flow 
    # is a record of all the nodes which were in the flow
    # note that a node is in its own total_flow
    # also, total_flow is a list, whereas the current_flow is a set
    dict_which_flow = dict()
    # for a pn in to_be_visited -> all the flows he is part of
    # ie a list of P_nodes, representing there flow
    # reciprocal of dict_flow 
    dict_end_of_flow = dict()
    # any pn -> where its flow converged back
    # this is what we want to build

    # ** Add a temporary global source **
    tmp_global_source_pn = P_node(main_graph=pg,main_target="tmp_source")
    tmp_global_source_pn.users = first_nodes = pg.first_nodes()
    for first_pn in first_nodes:
        first_pn.deps.add(tmp_global_source_pn)
    dict_nb_usages[tmp_global_source_pn] = len(first_nodes)

    # ** init **
    for pn in pg.list_nodes:
        if len(pn.users) == 0:
            to_be_visited.append(pn)
            dict_which_flow[pn] = set()

    # ** search **
    while to_be_visited != []:
        pn = to_be_visited.pop()
        current_flows = dict_which_flow[pn]
        continuing_flows = set([pn])
        dict_flow[pn] = set()
        dict_total_flow[pn] = [pn]
        # * check if end of flows *
        for flow_pn in current_flows:
            flow = dict_flow[flow_pn]
            flow.remove(pn)
            # equivalent to "seen.remove(n)" in cut_based_on_deps
            if flow == set():
                dict_end_of_flow[flow_pn] = pn
            else:
                continuing_flows.add(flow_pn)
        # * visit pn *
        for req_pn in pn.deps:
            # equivalent to seen.add(req_n) :
            for flow_pn in continuing_flows:
                tot_flow = dict_total_flow[flow_pn]
                flow     = dict_flow[flow_pn]
                flow.add(req_pn)
                if (not (req_pn is tmp_global_source_pn)
                and not (req_pn in tot_flow)):
                    tot_flow.append(req_pn)
            if req_pn in dict_which_flow:
                dict_which_flow[req_pn].update(continuing_flows)
            else:
                dict_which_flow[req_pn] = continuing_flows
            dict_nb_usages[req_pn]-=1
            if dict_nb_usages[req_pn]==0:
                to_be_visited.append(req_pn)

    # ** remove the temporary global source **
    for first_pn in first_nodes:
        first_pn.deps.remove(tmp_global_source_pn)

    # === SECOND ===
    # For each flow we have 4 options :
    # -> include the source or not
    # -> include the sink or not
    # But I will consider only 2 :
    # -> We always include the source
    # ATTENTION here sink/source are taken from .deps
    # relation perspective, e.g. outputs are sources.

    # Note that its merging here, not grouping


    all_options = set()
    # merge_flow_option set
    dict_options_pn_is_part_of = dict()
    # P_node -> merge_flow_option set
    # After each simplification, we actualize all 
    # the options which use some of the simplified nodes

    # ** init **
    for source_pn,sink_pn in dict_end_of_flow.items():
        flow = dict_total_flow[source_pn]
        flow.reverse()
        options = [merge_flow_option(flow)]
        if (not (sink_pn is tmp_global_source_pn) and len(flow)>2):
            flow_ = list(flow) ; flow_.remove(sink_pn)
            options.append(merge_flow_option(flow_))
        for opt in options:
            if len(opt.group) <= 1: continue
            all_options.add(opt)
            for pn in opt.group:
                if pn in dict_options_pn_is_part_of:
                    dict_options_pn_is_part_of[pn].add(opt)
                else:
                    dict_options_pn_is_part_of[pn] = set([opt])

    # for opt in all_options:
        # print(math.exp((config.merge_flow_value_fct(opt)+config.max_nodes_per_graph)/5))

    # ** main loop **
    while all_options != set():
        best_option = max(all_options,key=config.merge_flow_value_fct)
        all_options.remove(best_option)
        if config.merge_flow_stop_condition(pg,best_option):
            break
        else:
            best_group = list(best_option.group)
            new_pn = merge(best_group,pg)
            updated_opts = set()
            for pn in best_group:
                opts = list(dict_options_pn_is_part_of[pn])
                for opt in opts:
                    group = opt.group
                    # Case 1: one element of this group has already been replaced
                    if new_pn in group: 
                        group.remove(pn)
                        if len(group) < 2: # too small
                            all_options.discard(opt)
                    # Case 2: replace pn by new_pn
                    else:
                        group[group.index(pn)] = new_pn
                    if opt in all_options: updated_opts.add(opt)
                del dict_options_pn_is_part_of[pn]
            dict_options_pn_is_part_of[new_pn] = updated_opts
            for opt in updated_opts:
                opt.make_nb_nodes_and_nb_subnodes()
            print_debug(f"Successfully : {[ipn.name for ipn in best_group]} -> {new_pn.name}")

# ==========================



# =====================
# === Main function ===
# =====================

def S_to_P(sg : S_graph):
    pg = S_to_P_init(sg)
    rule_group_sequences(pg)
    rule_merge_small_flows(pg)
    pg.make_subgraph_id("0")
    pg.make_io_targets_attributes_of_subgraphs()
    pg.input_targets = [sg.init_node.main_target]
    pg.output_targets = [sg.hidden_output]
    return pg

# =====================



# ==========================
# === printing functions ===
# ==========================

color_leaf     = "blue"
color_subgraph = "blueviolet"
color_special  = "green"
color_edge     = color_leaf

def print_P_graph(pg : P_graph,name=None,open=True,render_format="svg"):
    # ----- init -----
    print(f"Partitioned forward graph : {len(pg.list_nodes)} nodes")
    if name is None: name = "Partitioned_forward_graph"
    dot = graphviz.Digraph(
        name,
        comment="P_graph = Partitioned forward graph")
    # ----- Core -----
    for input_pn in pg.input_nodes:
        dot.node(
            input_pn.name,
            f"INPUT {input_pn.main_target}",
            color=color_special, style="dashed")
    set_input_nodes = pg.input_nodes
    for pn in pg.list_nodes:
        if pn.is_leaf:
            dot.node(pn.name,pn.name,color=color_leaf)
        else:
            dot.node(
                pn.name,
                f"{pn.name}\n{pn.size()} nodes",
                color=color_subgraph)
        for req_pn in pn.deps:
            dot.edge(req_pn.name,pn.name,color=color_edge)
        for req_pn in pn.deps_global.intersection(set_input_nodes):
            dot.edge(req_pn.name,pn.name,color=color_special)
    # ----- render -----
    small_fcts.graph_render(dot,open,"P",render_format)
