# Parameter offload

The general direction is to start parameter offload with continuous assumption. Activation offload may follow if it's easy to support. Also, parameters offload should consider the possibility to optimize on CPU.

## parameter offload vs. activation offload

1. Parameters have only one source: prefetch. Thus $ofl+alive\geq 1$ is useful to control deletion.
2. Activations are assumed to be 0 by the end of bwd and start of fwd.
3. Activations(phantoms) highly depends on the sub-schedules chosen.
4. Activations memory occupation could come from different bwd; parameters should not share occupation without weight sharing.

## Formulation

T = #compute nodes

E = #edges from computation nodes to data nodes

L = #tensors occupy memory

We use indices $i,j$ to represent the computation step (i,j) during stage i, and $k$ for the variable contribution and $l$ for the tensor. $k\in l$ is used to describe that tensor $l$ includes contribution $k$.

### Variables

$Comp$ of shape [T, T], binary: $Comp(i,j) =1$ if compute j on stage i

$Alive$ of shape [T, T, E], continuous: $Alive(i,j,k)=1$ if variable k is alive at step (i,j)

<!-- $Ocp$ of shape [T, T, L], binary (can be continuous): whether tensor l occupies GPU memory -->

$time$ of shape [T, T], continuous: time spent during step (i,j)

$Prf$ of shape [T, T, E], continuous in [0,1]: fraction of variable k prefetched during step (i,j)

$Ofl$ of shape [T, T, E], continuous in [0,1]: fraction of variable k offloaded during step (i,j)

<!-- $PrfEnd$ of shape [T, T, E], binary: prefetch of variable is done during step (i,j)

$PrfProg$ of shape [T, E], continuous in [0,1]: prefetch progress of variable at the beginning of stage i. -->

### Objective

$$min.\sum_{i,j} time(i,j)$$

### Constraints

Time cost of each step is greater than the time cost of three channels
$$
time(i,j) \geq \sum_{l} (\sum_{k\in l} Ofl(i,j,k)*size(k))/bandwidthOfl\\
time(i,j) \geq \sum_{l} (\sum_{k\in l} Prf(i,j,k)*size(k))/bandwidthPrf\\
time(i,j) \geq Comp(i,j)*compCost(j)\\
$$

If no computation, no deletion/Offload/Prefetch

$$
Alive(i,j+1,k) \leq Alive(i,j,k)+Comp(i,j)\\
Alive(i,j+1,k) \geq Alive(i,j,k)-Comp(i,j)\\
Prf(i,j,k) \leq Comp(i,j)\\
Ofl(i,j,k) \leq Comp(i,j)
$$

To get a tensor, either compute the source or Prefetch from cpu
<!-- $$Alive(i,j+1,k) \leq Alive(i,j,k)+PrfEnd(i,j,k)\text{(if j is the src of k)}$$ -->
$$Alive(i,j+1,k) \leq Alive(i,j,k)+Prf(i,j,k)$$


Prefetch and Offload constraints
From bwd to fwd: weights are updated; offload increments monotonically;
$$
OflProg(i,j,k) = \sum^i_{bwd_k} Ofl(i,j,k)\\
% PrfProg(i,j,k) = \sum^i_{bwd_k} Prf(i,j,k)\\
OflProg(i,j,k) + Alive(i,j,k) \geq 1\\
Ofl(i,j,k) \leq Alive(i,j,k)\\
% PrfProg(i,j,k) \leq OflProg(i,j,k)\\ 
% PrfProg(k,k,k) + Alive(i,j,k) \geq 1\\ % Alive cannot be too low
% Can delete then prefetch again? 
% Prf(i,j,k) \leq \sum_{i'<i,j} Ofl(i',j,k)?\\
% PrfProg(i+1,k) = PrfProg(i,k) + \sum_j (Prf(i,j,k) - PrfEnd(i,j,k))\\
% 0 \leq PrfProg(i,k) + \sum_{j'<j} (Prf(i,j',k) - PrfEnd(i,j',k))\\
% 1 \geq PrfProg(i,k) + \sum_{j'<j} (Prf(i,j',k) - PrfEnd(i,j',k))\\
$$

Memory related constraints

$$
\forall i,j,k: \sum_{k} Alive(i,j,k)*weight_k+peak(k)*Comp(i,j,k)\leq mem_{avail}\\
mem_{avail} = M_{GPU} - 3*\sum_k (1-OflProg(k,k,k)) weight_k
$$
3 from Adam optimizer states (save for later)

<!-- $$
PrfProg(i,k) + \sum_{j'<j} (Prf(i,j',k) - PrfEnd(i,j',k)) \leq Ocp(i,j,l)\\
Alive(i,j,k) \leq Ocp(i,j,l), \forall k \in l\\
\sum_l Ocp(i,j,l) + overhead(j) * Comp(i,j) \leq Budget
$$ -->

## Limitations

1. Memory is counted for each step, thus no deletion happens in the middle of a step. Won't allow "Offload-delete-Prefetch" during one step.
