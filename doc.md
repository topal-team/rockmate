# Rockmate documentation

## API

### H_op
The operation types:
1. Run `H_C_node`: fast forward (thus HCN must be forward)
2. Del `H_D_node`: delete HDN
3. Run `H_sched`: run the corresponding fwd/bwd sched
4. Del `H_sched`: delete the phantoms passed from fwd to bwd



### H_sched
The H_sched correspond to one fwd/bwd schedule. When op_list is empty, 
it represents the bottom level fwd/bwd and information is assigned directly.
Note that a `H_sched` always represents fwd/bwd schedule thus there is no fwd_sched.

`.op_list`: `list` of `H_op`.
`.hgraph`: `H_sched` is the schedule to run one `H_graph`
`.fwd/bwd_time`: time of fwd/bwd
`.fwd/bwd_overhead`: overhead of fwd/bwd
`.phantoms`: set of `H_D_node` or `H_sched`. saved from fwd to bwd. `H_sched` represents the phantoms of the sub_graph corresponds to the schedule.
`.mem`: memory of phantoms.

### H_graph

`H_graph` is the high level structure to represent the (sub)graph of the network. Two types of nodes can be inside a `H_graph`: 

`H_C_node`'s: represent the computation of certain sub-parts of the network; `H_D_node`'s: represent data generated from `H_C_node`. Note that there are `H_D_node`'s to represent the input/output data, even though these tensors can be represented by other `H_D_node` in the higher level.

The smallest `H_graph` represents only one torch function, and it contains no H_D_node.

### H_C_node

When `H_C_node` represents the part that requires gradient, it pairs with another `H_C_node` to make a fwd/bwd pair. Both fwd/bwd `H_C_node` share the **same** `.sub_graph` which represents the `H_graph` for the subset. If the `H_C_node.is_leaf=True`, `sub_graph` has no nodes. 

A forward `H_C_node` has `.is_fwd=True` (sometimes such `H_C_node`'s have no `sub_graph`). And a forward node may not be executed for backward (we call it fast forward). In practice, this can be done in `with torch.no_grad()`. Since this execution approach is not for backward, the information is not stored in `.sub_graph` but in `H_C_node` itself: `.fwd_time`, `fwd_overhead` and `ff_op_list` represent the fast forward information, which is `empty` or 0 for backward `H_C_node`.

### H_D_node

Future work: `H_D_node` should be allowed to represent a set of torch tensors! It is not for the concern of optimality, but for the efficiency: we may want to group some data tensors to reduce the complexity of ILP. Memory cost: one alive, cost all; dependency control: to run the user, all need to be alive. Sure there can be a waste, that's the price of reducing complexity.