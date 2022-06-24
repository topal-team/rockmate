#!/usr/bin/python

from .sequence import *

try:
    import dynamic_programs as dp
    c_version_present = True
except ImportError:
    c_version_present = False


def floating_index_in_array(L, m, s, t, l):
    m_factor = (L+1) * (L+2) * (2*L+6) // 12;
    i = L - s
    j = t - s
    k = l - t
    return m * m_factor + i*(i+1)*(2*i+4)//12 + (i+1)*j - j*(j-1)//2 + k



def compute_table(chain, mmax):
    """Returns the optimal table: a tuple containing: 
    Opt[m][s][t][l]
        with s = 0...chain.length
        and  t = s...chain.length 
        and  l = s+1..chain.length
        and  m = 0...mmax
    what[m][lmin][lmax] is (True,) if the optimal choice is a chain checkpoint
                           (False, s', r, t') if the optimal choice is a leaf checkpoint with params s' r t'
    The computation uses dynamic programming"""
    
    fw = chain.fweigth + [0] ## forward time
    bw = chain.bweigth ## backward time, not used
    cw = chain.cweigth  + [0]## size of x (and of y)
    cbw = chain.cbweigth + [0] ## size of xbar
    fwd_tmp = chain.fwd_tmp + [0]
    bwd_tmp = chain.bwd_tmp + [0]
    
    # Build table
    L = chain.length
    infty = float("inf")
    opt = [ {} for m in range(mmax+1) ]
    what = [ {} for m in range(mmax+1) ]
    # # Let's waste some space. Does not look like numpy can do triangular arrays
    # L = chain.length
    # opt = np.empty([L+1, L+1, L+1, mmax+1])
    # what = np.empty([L+1, L+1, L+1, mmax+1, 3], dtype=int16)

    def add_value(m, s, t, l, value, decision = None):
        opt[m][(s, t, l)] = value
        if decision:
            what[m][(s, t, l)] = decision
        
    

    # Precomputation: partialSumsFW[i] = sum(fw[j] for j in range(i))
    partialSumsFW = []
    value = 0
    for v in fw:
        partialSumsFW.append(value)
        value += v
    partialSumsFW.append(value)

    # Initialize borders of the tables 
    for m in range(mmax + 1):
        for i in range(chain.length + 1):
            limit = max(cw[i+1] + cbw[i+1] + fwd_tmp[i],
                        cw[i]+ cw[i+1] + cbw[i+1] + bwd_tmp[i])
            if m >= limit: ## Equation (1)
                add_value(m, i, i, i, fw[i] + bw[i])
            else:
                add_value(m, i, i, i, infty)



    def memNull(s, l):
        first = cw[l+1] + cw[s+1] + fwd_tmp[s]
        if l <= s+1: return first
        second = cw[l+1] + max(cw[j] + cw[j+1] + fwd_tmp[j] for j in range(s+1, l))
        return max(first, second)

    def memAll(s, l):
        return max(cw[l+1] + cbw[s+1] + fwd_tmp[s],
                   cw[s] + cw[s+1] + cbw[s+1] + bwd_tmp[s])
    
    # Compute everything
    for m in range(mmax + 1):
        for d in range(1, L+1): ## d is l-s
            for s in range(L+1 - d):
                l = s + d
                for t in range(s, l+1):
                    if s == t and m >= memAll(s, l):
                        chain_checkpoint = opt[m][(s, s, s)] + opt[m-cbw[s+1]][(s+1, s+1, l)]
                    else: chain_checkpoint = infty
                    def leaf_chckpt(sp, r, tp):
                        try: 
                            mp = m - cw[r] + cw[s]
                            assert mp >= 0
                            if mp < cw[sp]: return infty
                            ## was: sum(fw[s:sp])
                            return ( partialSumsFW[sp] - partialSumsFW[s] + opt[mp - cw[sp]][(sp, tp, l)]
                                     + opt[mp][(r, t, tp - 1)] )
                        except KeyError:
                            print("Failed to compute leaf_chckpt for", s, t, l, "with", sp, r, tp, mp)
                            raise
                    if m >= memNull(s, l):
                        ## TODO: There may be a way to avoid recomputing some of these triplets.
                        ##   leaf_chckpt(...) only depends on l and t, not on s.
                        leaf_checkpoints = ( ((sp, r, tp), leaf_chckpt(sp, r, tp))
                                             for r in range(s, t+1) if cw[s] <= cw[r]
                                             for tp in range(t+1, l+1)
                                             for sp in range(r+1,  tp+1) )
                        try:
                            best_leaf = min(leaf_checkpoints, key = lambda t: t[1])
                        except ValueError:
                            best_leaf = None
                    else: best_leaf = None

                    if best_leaf and best_leaf[1] != infty and best_leaf[1] <= chain_checkpoint:
                        add_value(m, s, t, l, best_leaf[1],
                                  (False, best_leaf[0]))
                    else: 
                        add_value(m, s, t, l, chain_checkpoint, (True,))

    return (opt, what)


# Computes the optimal sequence, recursive helper function
def floating_rec(chain, s, t, l, cmem, opt_table):
    if cmem < 0:
        raise ValueError("Can not process a chain with negative memory {cmem}".format(cmem=cmem))
    opt, what = opt_table
    sequence = Sequence(Function("Floating", s, t, l, cmem))
    if opt[cmem][(s, t, l)] == float("inf"):
        raise ValueError("Can not process this chain from index {s} to {l} throught {t} with memory {cmem}".format(s=s, t=t, l=l, cmem=cmem))
    if s == l:
        if s == chain.length:
            sequence.insert(Loss())
        else: 
            sequence.insert(ForwardEnable(s))
            sequence.insert(Backward(s))
        return sequence
    
    if what[cmem][(s, t, l)][0]:
        assert s == t
        sequence.insert(ForwardEnable(s))
        sequence.insert_sequence(floating_rec(chain, s+1, t+1, l, cmem - chain.cbweigth[s+1], opt_table))
        sequence.insert(Backward(s))
    else:
        (sp, r, tp) = what[cmem][(s, t, l)][1]
        if r != s: 
            sequence.insert(ForwardNograd(s))
        else: 
            sequence.insert(ForwardCheck(s))
        for k in range(s+1, sp):
            if k != r: sequence.insert(ForwardNograd(k))
            else: sequence.insert(ForwardCheck(k))
        mp = cmem - chain.cweigth[r] + chain.cweigth[s]
        sequence.insert_sequence(floating_rec(chain, sp, tp, l, mp - chain.cweigth[sp], opt_table))
        sequence.insert_sequence(floating_rec(chain, r, t, tp-1, mp, opt_table))
    return sequence
                        

# Computes the optimal sequence for the given parameters
def floating(chain, memory_limit, force_python = False):
    cmem = memory_limit - chain.cweigth[0]
    if c_version_present and not force_python:
        opt_table = dp.floating_compute_table(chain, cmem)
    else: 
        opt_table = compute_table(chain, cmem)

    return floating_rec(chain, 0, 0, chain.length, cmem, opt_table)
