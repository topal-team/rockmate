# ==========================
# modified version of rotor algo
# based on rotor/algorithms/persistent.py
# ==========================

from .utils import *
from .defs import RK_chain

def compute_table(chain : RK_chain, mmax):
    """Returns the optimal table:
    Opt[m][lmin][lmax] : int matrix
        with lmin = 0...chain.length
        and  lmax = lmin...chain.length (lmax is not included)
        and  m    = 0...mmax
    What[m][lmin][lmax] is :
        (True, k) if the optimal choice is a chain chkpt
        -> ie F_e this block with solution k
        (False,j) if the optimal choice is a leaf  chkpt
        -> ie F_c and then F_n (j-1) blocks
    """

    ln      = chain.ln
    fw      = chain.fw
    bw      = chain.bw
    cw      = chain.cw
    cbw     = chain.cbw
    fwd_tmp = chain.fwd_tmp
    bwd_tmp = chain.bwd_tmp
    ff_fwd_tmp = chain.ff_fwd_tmp
    ff_fw   = chain.ff_fw
    nb_sol  = chain.nb_sol

    opt =  [[{} for _ in range(chain.length+1)] for _ in range(mmax + 1)]
    what = [[{} for _ in range(chain.length+1)] for _ in range(mmax + 1)]
    # -> Last one is a dict because its indices go from i to l. 
    # -> Renumbering will wait for C implementation

    # -- Initialize borders of the tables for lmax-lmin = 0 --
    for m in range(mmax + 1):
        for i in range(ln + 1):
            # lmax = lmin = i
            possibilities = []
            for k in range(nb_sol[i]):
                limit = max(cw[i+1] + cbw[i+1][k] + fwd_tmp[i][k],
                            cw[i]+ cw[i+1] + cbw[i+1][k] + bwd_tmp[i][k])
                if m >= limit:
                    possibilities.append(k,fw[i][k] + bw[i][k])
            if possibilities == []:
                opt[m][i][i] = float("inf")
            else:
                best_sol = min(possibilities, key = lambda t: t[1])
                opt[m][i][i] = best_sol[1]
                what[m][i][i] = (True,best_sol[0])

    # -- dynamic program --
    for m in range(mmax + 1):
        for d in range(1, ln + 1):
            for a in range(ln + 1 - d):
                b = a + d
                # lmin = a ; lmax = b
                mmin = cw[b+1] + cw[a+1] + ff_fwd_tmp[a]
                if b > a+1:
                    mmin = max(mmin,
                        cw[b+1] + max(cw[j]+cw[j+1]+ff_fwd_tmp[j]
                        for j in range(a+1, b)))
                if m < mmin:
                    opt[m][a][b] = float("inf")
                else:
                    # -- Solution 1 --
                    sols_later = [
                        (j,(sum(ff_fw[a:j])
                            + opt[m-cw[j]][j][b]
                            + opt[m][a][j-1]))
                        for j in range(a+1, b+1)
                        if m >= cw[j] ]
                    if sols_later:
                        best_later = min(sols_later, key = lambda t: t[1])
                    else: best_later = None

                    # -- Solution 2 --
                    # -> we can no longer use opt[i][i] because the cbw
                    # -> now depend on the F_e chosen. 
                    sols_now = []
                    for k in range(nb_sol[a]):
                        mem_f = cw[a+1] + cbw[a+1][k] + fwd_tmp[a][k]
                        mem_b = cw[a]+cw[a+1]+cbw[a+1][k]+bwd_tmp[a][k]
                        limit = max(mem_f,mem_b)
                        if m >= limit:
                            time = fw[a][k] + bw[a][k]
                            time += opt[m-cbw[a+1][k]][a+1][b]
                            sols_now.append((k,time))
                    if sols_now:
                        best_now = min(sols_now, key = lambda t: t[1])
                    else: best_now = None

                    # -- best of 1 and 2 --
                    if best_later is None and best_now is None:
                        opt[m][a][b] = float("inf")
                    elif best_later is None or best_now[1]<best_later[1]:
                        opt[m][a][b] = best_new[1]
                        what[m][a][b] = (True, best_new[0])
                    else:
                        opt[m][a][b] = best_later[1]
                        what[m][a][b] = (False, best_later[0])

    return (opt,what)


##########################################
############ TODO ###################


def persistent_rec(chain, lmin, lmax, cmem, opt_table, print_mem=False):
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
    if print_mem:
        print(lmin,cmem)
    if what[cmem][lmin][lmax][0]:
        ind = what[cmem][lmin][lmax][1]
        sequence.insert(ForwardEnable(lmin,ind))
        #sequence.insert_sequence(persistent_rec(chain, lmin+1, lmax, cmem - chain.cweight[lmin+1], opt_table))
        sequence.insert_sequence(persistent_rec(chain, lmin+1, lmax, cmem - chain.m2t[lmin+1][ind][0], opt_table,print_mem=print_mem))
        sequence.insert(Backward(lmin))
    else:
        j = what[cmem][lmin][lmax][1]
        sequence.insert(ForwardCheck(lmin))
        for k in range(lmin+1, j): sequence.insert(ForwardNograd(k))
        sequence.insert_sequence(persistent_rec(chain, j, lmax, cmem - chain.cweight[j], opt_table,print_mem=print_mem))
        sequence.insert_sequence(persistent_rec(chain, lmin, j-1, cmem, opt_table))
    return sequence

