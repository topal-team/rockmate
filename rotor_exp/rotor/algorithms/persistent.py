#!/usr/bin/python

from .sequence import *

try:
    import dynamic_programs as dp
    c_version_present = True
except ImportError:
    c_version_present = False


def compute_table(chain, mmax):
    """Returns the optimal table: a tuple containing: 
    Opt[m][lmin][lmax] with lmin = 0...chain.length
         and lmax = lmin...chain.length (lmax is not included) and m = 0...mmax
    what[m][lmin][lmax] is (True,) if the optimal choice is a chain checkpoint
                           (False, j) if the optimal choice is a leaf checkpoint of length j
    The computation uses dynamic programming"""
    
    fw = chain.fweigth + [0] ## forward time
    bw = chain.bweigth ## backward time, not used
    cw = chain.cweigth  + [0]## size of x (and of y)
    cbw = chain.cbweigth + [0] ## size of xbar
    fwd_tmp = chain.fwd_tmp + [0]
    bwd_tmp = chain.bwd_tmp + [0]
    
    # Build table
    opt = [[{} for _ in range(chain.length+1)] for _ in range(mmax + 1)]
    what = [[{} for _ in range(chain.length+1)] for _ in range(mmax + 1)]
    ## Last one is a dict because its indices go from i to l. Renumbering will wait for C implementation
    
    # Initialize borders of the tables for lmax-lmin = 0
    for m in range(mmax + 1):
        for i in range(chain.length + 1):
            #lmax-lmin = 0
            limit = max(cw[i+1] + cbw[i+1] + fwd_tmp[i],
                        cw[i]+ cw[i+1] + cbw[i+1] + bwd_tmp[i])
            if m >= limit: ## Equation (1)
                opt[m][i][i] = fw[i] + bw[i]        
            else:
                opt[m][i][i] = float("inf")

    # Compute everything
    for m in range(mmax + 1):
        for d in range(1, chain.length + 1): 
            for i in range(chain.length+1 - d):
                # for l in range(i+1, chain.length + 1):
                l = i + d
                mmin = cw[l+1] + cw[i+1] + fwd_tmp[i]
                if l > i+1:
                    mmin = max(mmin, cw[l+1] + max(cw[j] + cw[j+1] + fwd_tmp[j] for j in range(i+1, l)))
                if m < mmin:
                    opt[m][i][l] = float("inf")
                else:
                    leaf_checkpoints = [(j, sum(fw[i:j]) + opt[m-cw[j]][j][l] + opt[m][i][j-1]) for j in range(i+1, l+1) if m >= cw[j]]
                    if leaf_checkpoints:
                        best_leaf = min(leaf_checkpoints, key = lambda t: t[1])
                    else: best_leaf = None
                    if m >= cbw[i+1]: 
                        chain_checkpoint = opt[m][i][i] + opt[m - cbw[i+1]][i+1][l]
                    else: chain_checkpoint = float("inf")
                    if best_leaf and best_leaf[1] <= chain_checkpoint:
                        opt[m][i][l] = best_leaf[1]
                        what[m][i][l] = (False, best_leaf[0])
                    else: 
                        opt[m][i][l] = chain_checkpoint
                        what[m][i][l] = (True,)
    return (opt, what)


# Computes the optimal sequence, recursive helper function
def persistent_rec(chain, lmin, lmax, cmem, opt_table):
    """ chain : the class describing the AC graph
        lmin : index of the first forward to execute
        lmax : upper bound index of the last forward to execute (not included)
        cmem : number of available memory slots
        Return the optimal sequence of makespan Opt_hete[cmem][lmin][lmax-lmin]"""
    if cmem <= 0:
        raise ValueError("Can not process a chain with negative memory {cmem}".format(cmem=cmem))
    opt, what = opt_table
    sequence = Sequence(Function("Persistent", lmax-lmin, cmem))
    if opt[cmem][lmin][lmax] == float("inf"):
        raise ValueError("Can not process this chain from index {lmin} to {lmax} with memory {cmem}".format(lmin=lmin, lmax=lmax, cmem=cmem))
    if lmin == lmax:
        if lmin == chain.length:
            sequence.insert(Loss())
        else: 
            sequence.insert(ForwardEnable(lmin))
            sequence.insert(Backward(lmin))
        return sequence
    
    if what[cmem][lmin][lmax][0]:
        sequence.insert(ForwardEnable(lmin))
        sequence.insert_sequence(persistent_rec(chain, lmin+1, lmax, cmem - chain.cbweigth[lmin+1], opt_table))
        sequence.insert(Backward(lmin))
    else:
        j = what[cmem][lmin][lmax][1]
        sequence.insert(ForwardCheck(lmin))
        for k in range(lmin+1, j): sequence.insert(ForwardNograd(k))
        sequence.insert_sequence(persistent_rec(chain, j, lmax, cmem - chain.cweigth[j], opt_table))
        sequence.insert_sequence(persistent_rec(chain, lmin, j-1, cmem, opt_table))
    return sequence
                        
# Computes the optimal sequence for the given parameters
def persistent(chain, memory_limit, force_python = False):
    if c_version_present and not force_python:
        opt_table = dp.persistent_compute_table(chain, memory_limit)
    else: 
        opt_table = compute_table(chain, memory_limit)
    return persistent_rec(chain, 0, chain.length, memory_limit - chain.cweigth[0], opt_table)


if __name__ == '__main__':
    from .utils import simulate_sequence
    from . import parameters
    
    params = parameters.parse_arguments()
    start_time = time.time()
    seq = persistent(params)
    end_time = time.time()
    print(seq)
    maxUsage = simulate_sequence(seq, params.chain.length, chain=params.chain)
    print("Chain length: ", params.chain.length)
    print("Sequence: ", seq)
    print("Memory: ", maxUsage)
    print("Makespan: ", seq.makespan)
    print("Compute time: %.3f ms"%(1000*(end_time - start_time)))
    

