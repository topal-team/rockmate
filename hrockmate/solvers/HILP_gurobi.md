
# ILP model for Hierarchical Rockmate solver

**Context:** We assume that we have an arbitrary graph $H$, where each
node `hcn` is an operation. We have $T$ compute nodes, $I$ data nodes.
Compute nodes are numbered in a topological order. 

The novelty of H-rockmate is that each compute node can be computed
with one of several options, indexed by $o$. There is a connection
between the forward and the respective backward computations, and each
backward should be performed with the same option as its corresponding
forward. A pair of a forward and the corresponding backward nodes are
called a *layer*.

In addition, we introduce the concept of *phantom nodes*, which represent
the data saved in memory between a forward and its corresponding
backward. They are like special data nodes, with two characteristics:
  + their only dependencies are with the FWD and corresponding BWD:
    created by FWD, consumed by BWD, not needed by any other computing
    node.
  + the values saved in a phantom node (and thus the associated memory
    size) depend on the option used for the FWD and BWD
    computations. For this reason, a phantom node has several options.

In this formulation, we consider schedules in which for a given
phantom node, only one option is present in memory at a given
time. However, it is possible that a phantom node is produced several
times with different options during the course of the schedule.

An output value of the forward compute node (ie, a data node which is
computed during forward and used by another compute node) is never
included in the phantom node. However, it happens that an output value is
also used within the forward computation to produce other results. An
example could be:
```python
def layer(a):
	x = f(a)
	y = g(x)
	return x, y
```

In this example, the value `x` is both an output of the layer and used
to produce `y`. In that case, the backward schedule might choose
either to use `x` as input to be able to perform the backward of `g`
(if having it in memory between forward and backward fits in the
budget), or to recompute it during backward. The implication is that
for a given layer, different options might have different input
dependencies for the backward compute node. Additional dependencies
are stored in the `dep_interfaces_data` field of each schedule.

Just like in rk-checkmate, a data node can have several
predecessors. This happens in backward when computing gradients: each
computation is a contribution to the same memory slot (gradients are
accumulated). A successor of such a data node can only be processed if
all the contributions have been computed.

**Intuition:** Like Checkmate, the schedule is divided into $T$
phases. The goal of phase $t$ is to compute node $t$ for the first
time. 

**Notations** Compute nodes are denoted with index $k$, data nodes
with index $d$, options with index $o$, phases with index $t$ and
layers (a pair of forward and corresponding backward nodes) with index
$j$.

**Variables:**
+ $R_{k, t, o}$ is $1$ if and only if node $k$ is computed with option
$o$ during phase $t$.
+ $P_{t, d}$ is $1$ if and only if data node $d$ is present in memory
  before phase $t$.
+ $S_{k, t, d}$ for $k$ predecessor of $d$ is $1$ if and only if the
  contribution of compute node $k$ has been included in data node $d$
  before phase $t$.
+ $Sp_{j, t, o}$ is $1$ if and only if the phantom of layer $j$ is
  saved with option $o$ before phase $t$.
+ $C_{k, t, d}$ is $1$ iff data node $d$ is created when computing
  node $k$ during phase $t$
+ $D_{k, t, d}$ is $1$ iff data node $d$ is deleted after computing
  node $k$ during phase $t$

Some layers do not have a backward computation, I am not sure which case this represents

**Formulation:** Minimize total running time, \ie sum of $R_{i, t,
o}*\text{time of computing option $o$ for node $i$}$

**Ordering constraints**
* Node $i > t$ can not be computed in phase $t$
* You can not save any phantom $j$ from node $i > t$ in phase $t$
* You can not save the result of compute node $k$ in any phase $t \leq k$
* Data node $i$ cannot be in memory in any phase before the phase of its predecessor with smallest index

**Validity constraints**
* In the last stage, every source edge of `input_grad` (the gradient of the input) should be alive or executed
* Forward start with no phantoms
* in the end of bwd, del every phantoms
* In a given phase, a node is computed with only one option, and only
  one phantom is in memory (THERE IS PROBABLY A PROOF THAT THIS IS
  OPTIMAL? If you keep two options, you can just forget the one that
  is most expensive to compute... Not clear, but nvm.)
* Node $t$ is executed in phase $t$
* Loss node is executed only once

**Data dependencies** 
* For any create edge $(k, d)$, if contribution $k$ exists before
  phase $t$, then data node $d$ is in memory before phase $t$.
* A contribution can be present after phase $t$ only if it was present
  before or it is computed during
* Computing a successor of $k$ during phase $t$ requires either
  computing $k$ during phase $t$ or having it saved before

**Constraints about phantom nodes**
* For each layer, and for each option:
  * A phantom can be saved after phase $t$ only if it was saved before or the FWD is computed during $t$
  * If a phantom was saved before phase $t$ and its BWD was not
    computed during $t$, then it is still saved after $t$
  * Computing a BWD during phase $t$ requires either having the
    phantom saved before $t$ or computing the FWD during $t$
  * for any edge (k, i) in create_list that contributes to an input of
	 the backward of this option: computing the BWD during phase $t$
	 can only be done if some option of $k$ is computed during $t$ or
	 its contribution is stored before $t$

**Alive status of values**
* At least 0, at most 1 version of node $i$ is alive at a time
* If node $k$ is computed at phase $t$, then either data node $i$ is alive after node $k$, or it is deleted.
* You can only create a value if you compute the corresponding node
* A data node is alive after phase $t$ iff it is alive after the last computing node that produces it
* No data node is alive after the last phase $T$
* A data node is deleted after computing node $k$ if it is not used afterwards -- requires 'OR' construction cf checkmate

**Memory constraint**
* All on expressions `U`, very close to Checkmate
  * `U[t, 0]` is based on `P(t, .)` plus what is computed for node 0
  * `U[t, k+1]` is based on `U[t, k]`
* Constraint: peak for computing node $k$ is at most `peak_budget`
  * peak is `U[t, k]` plus overhead of computing node $k$ plus the
    amount of deleted memory after computing node $k$
	Because we assume that deleted values were alive during the peak
* Save constraint: for all computation nodes during the loss phase,
  the memory usage should be at most `save_budget`

**Correction terms**

The goal of this is to take into account what happens *during* the
computation of node $k$. The peak might depend on whether some values
are alive before or after computing node $k$. Example: if node $k$
deletes a value in the middle of computation, its peak assumes that
the value disappears. If in our schedule we need that value later for
something else, we will keep it, which may or may not change the peak.

Given a phase $t$, and an option (and thus a schedule) for node
$k$. For simplicity of presentation, let us consider only inputs; the
situation with outputs is similar and symmetric. We start with the
highest possible memory usage, denoted $O$, which assumes that no
input is deleted after the computation of this schedule of node
$k$. Consider a substep $i$ of the schedule: its *actual* memory usage
is $O - \sum_{inp \in \text{inputs not needed after $i$}} \text{$inp$
is deleted after node $k$}\cdot \text{size of $inp$}$.

In the formulation, variable $D_{k, t, d}$ is equal to $1$ iff a value
$d$ is deleted after computing node $k$, so that this actual memory
usage can be expressed as a linear expression. We can thus write such
a constraint for each substep of the schedule, and this provides a
more precise assessment of the memory usage of the solution.  The case
of output values is the same, except that we care whether the output
value is created during the computation of node $k$, which is
represented with variable $C_{k, t, d}$.

An interesting remark is that it is not necessary to write one
constraint for each substep: if the set of inputs not needed after
substeps $i$ and $j$ are the same, we can keep only one of both
constraints (the one with larger overhead $O$). The number of
constraints is thus bounded by $\min(\text{schedule length},
2^{|\{\text{inputs}\}| + |\{\text{outputs}\}|})$. In practice, the
number of different constraints remain low.


# Python variable documentation
* `gcd`: normalization factor for memory usage
* `sub_gs`: subgraphs (=layers), each graph only appears once (so that if there are duplicates, we only include them once)
  + duplicates only happen for forward and backward of the same computation
  + every subgraph is shared between its FWD and BWD nodes
* `hcn2sub_g`: list of length $N$, where $N$ is the number of nodes in
  the graph $H$. Each element of this list is an index in `sub_gs`, to
  reference the subgraph of node $i$.
* `sub_g2hcn`: reverse indexing, contains the list of nodes that use a given subgraph (ie, the FWD and the BWD)
* `nOpts`: list, entry $i$ is the number of options of node $i$
* `nR`: same as nOpts, contains number of recompute possibilities?
	`nR` is equal to `nOpts` for a bwd operation, and to `nOpts+1` for
	a fwd operation.
* `time` and `overhead` are lists of lists, the $i$-th list has length
  `nR[i]`
* `saved_mem` (list of lists) contains the saved memory size of each option of each subgraph
* `T`: number of compute nodes
* `I`: number of data nodes
* `J`: number of subgraphs

* `input_grad_indices`: indices of the data nodes
* `_deps_d`: list of length $I$, containing the list of indices (in
  `hgraph.list_hcn`) of the compute nodes that are predecessors of
  each data node
	
* `_users_d`: list of length $I$, containing the list of indices (in
  `hgraph.list_hcn`) of the successors of each data node
* `_users_c`: list of length $T$, containing the list of indices (in
  `hgraph.list_hdn`) of the successors of each compute node
  
* `create_list`: list of edges, as pairs $(k, i)$, from compute node $k$ to its
  users $i$. Is the flattened version of `users_c`. Length is `Cr`
* `delete_list`: list of edges $(k, i)$, where $i$ spans all data
  nodes, and $k$ spans all its predecessors and users. Length is
  `De`. It is the list of data nodes that might be deleted when
  computing node $k$.
  + This also contains predecessors of data node $i$ (`_deps_d[i]`): a
    value can be deleted when computing a node that creates it, for
    example because the computation creates two values, and only one
    of them is useful next.

# Gurobi variables
* `R`: list of dicts of length $T$, each has length `T * nR[i]`
  + `R[i][t, o] == 1` iff operation $i$ is computed with option $o$ during phase $t$

* `Sp`: list of dicts of length $J$, each contains $(T+1) * (\text{number of options of subgraph})$
  + `Sp[j][t, o] == 1` iff you save the phantom of option $o$ of subgraph $j$ before phase $t$

* `S`: dict with $T*Cr$ values. `S[t][j] == 1` for the $j$-th edge $(k, i)$ iff the contribution of compute node $k$ to data node $i$ has been included in data node $i$ before phase $t$
* `P`: dict with $T*I$ values. `P[t][d] == 1` iff data node $d$ is present in memory before phase $t$ 
* `create`, `delete`: dicts with $T*Cr$ and $T*De$ values
  * `create[t, e]` for edge $e=(k, i)$ is 1 iff data node $i$ is created when computing node $k$ during phase $t$
  * `delete[t, e]` for edge $e=(k, i)$ is 1 iff data node $i$ is deleted when computing node $k$ during phase $t$

# Expressions
* `sumR`: dict with $T*T$ values, containing $\sum_o R[\cdot][\cdot, o]$
* `sumSp`: dict with $J*(T+1)$ values, containing $\sum_o Sp[\cdot][\cdot, o]$
* `alive`: dictionary indexed by $(t, k, i)$: phase $t$, edge $(k, i)$
  + equals 1 iff data node $i$ alive after computing node $k$ in phase $t$
* `U`: dictionary indexed by phase $t$ and compute node $k$
  + equals memory usage after computing $k$ in phase $t$
