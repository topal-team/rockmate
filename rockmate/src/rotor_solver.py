# ==========================
# modified version of rotor algo
# contains RK_Sequence builder -> depends on RK_Chain
# based on rotor/algorithms/persistent.py
# ==========================

from rkgb.utils import print_debug
from rockmate.def_chain import RK_Chain
from rockmate.def_sequence import (
    SeqBlockFn,
    SeqBlockFc,
    SeqBlockFe,
    SeqBlockBwd,
    SeqLoss,
    RK_Sequence,
)

# ==========================
# ==== DYNAMIC PROGRAM =====
# ==========================


def psolve_dp_functionnal(chain, mmax, opt_table=None):
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

    ln = chain.ln
    fw = chain.fw
    bw = chain.bw
    cw = chain.cw
    cbw = chain.cbw
    fwd_tmp = chain.fwd_tmp
    bwd_tmp = chain.bwd_tmp
    ff_fwd_tmp = chain.ff_fwd_tmp
    ff_fw = chain.ff_fw
    nb_sol = chain.nb_sol

    opt, what = opt_table if opt_table is not None else ({}, {})
    # opt = dict()
    # what = dict()

    def opt_add(m, a, b, time):
        if not m in opt:
            opt[m] = dict()
        if not a in opt[m]:
            opt[m][a] = dict()
        opt[m][a][b] = time

    def what_add(m, a, b, time):
        if not m in what:
            what[m] = dict()
        if not a in what[m]:
            what[m][a] = dict()
        what[m][a][b] = time

    # -> Last one is a dict because its indices go from i to l.
    # -> Renumbering will wait for C implementation

    # -- Initialize borders of the tables for lmax-lmin = 0 --
    def case_d_0(m, i):
        possibilities = []
        for k in range(nb_sol[i]):
            limit = max(
                cw[i] + cbw[i + 1][k] + fwd_tmp[i][k],
                cw[i] + cbw[i + 1][k] + bwd_tmp[i][k],
            )
            if m >= limit:
                possibilities.append((k, fw[i][k] + bw[i][k]))
        if possibilities == []:
            opt_add(m, i, i, float("inf"))
        else:
            best_sol = min(possibilities, key=lambda t: t[1])
            opt_add(m, i, i, best_sol[1])
            what_add(m, i, i, (True, best_sol[0]))
        return opt[m][i][i]

    # -- dynamic program --
    nb_call = 0

    def solve_aux(m, a, b):
        if (m not in opt) or (a not in opt[m]) or (b not in opt[m][a]):
            nonlocal nb_call
            nb_call += 1
            if a == b:
                return case_d_0(m, a)
            #  lmin = a ; lmax = b
            mmin = cw[b + 1] + cw[a + 1] + ff_fwd_tmp[a]
            if b > a + 1:
                mmin = max(
                    mmin,
                    cw[b + 1]
                    + max(
                        cw[j] + cw[j + 1] + ff_fwd_tmp[j]
                        for j in range(a + 1, b)
                    ),
                )
            if m < mmin:
                opt_add(m, a, b, float("inf"))
            else:
                #  -- Solution 1 --
                sols_later = [
                    (
                        j,
                        (
                            sum(ff_fw[a:j])
                            + solve_aux(m - cw[j], j, b)
                            + solve_aux(m, a, j - 1)
                        ),
                    )
                    for j in range(a + 1, b + 1)
                    if m >= cw[j]
                ]
                if sols_later:
                    best_later = min(sols_later, key=lambda t: t[1])
                else:
                    best_later = None

                #  -- Solution 2 --
                # -> we can no longer use opt[i][i] because the cbw
                # -> now depend on the F_e chosen.
                sols_now = []
                for k in range(nb_sol[a]):
                    mem_f = cw[a + 1] + cbw[a + 1][k] + fwd_tmp[a][k]
                    mem_b = cw[a] + cbw[a + 1][k] + bwd_tmp[a][k]
                    limit = max(mem_f, mem_b)
                    if m >= limit:
                        time = fw[a][k] + bw[a][k]
                        time += solve_aux(m - cbw[a + 1][k], a + 1, b)
                        sols_now.append((k, time))
                if sols_now:
                    best_now = min(sols_now, key=lambda t: t[1])
                else:
                    best_now = None

                # -- best of 1 and 2 --
                if best_later is None and best_now is None:
                    opt_add(m, a, b, float("inf"))
                elif best_later is None or (
                    best_now is not None and best_now[1] < best_later[1]
                ):
                    opt_add(m, a, b, best_now[1])
                    what_add(m, a, b, (True, best_now[0]))
                else:
                    opt_add(m, a, b, best_later[1])
                    what_add(m, a, b, (False, best_later[0]))
        return opt[m][a][b]

    solve_aux(mmax, 0, ln)

    print_debug(f"Nb calls : {nb_call}")
    return (opt, what)


# ==========================
#  ==== SEQUENCE BUILDER ====
# ==========================


def pseq_builder(chain, memory_limit, opt_table):
    # returns :
    # - the optimal sequence of computation using mem-persistent algo
    mmax = memory_limit - chain.cw[0]
    # opt, what = solve_dp_functionnal(chain, mmax, *opt_table)
    opt, what = opt_table
    #  ~~~~~~~~~~~~~~~~~~
    def seq_builder_rec(lmin, lmax, cmem):
        seq = RK_Sequence()
        if lmin > lmax:
            return seq
        if cmem <= 0:
            raise ValueError(
                "Can't find a feasible sequence with the given budget"
            )
        if opt[cmem][lmin][lmax] == float("inf"):
            """
            print('a')
            print(chain.cw)
            print('abar')
            for i in range(chain.ln):
                print(chain.cbw[i])
                print(chain.fwd_tmp[i])
                print(chain.bwd_tmp[i])
            """
            raise ValueError(
                "Can't find a feasible sequence with the given budget"
                # f"Can't process this chain from index "
                # f"{lmin} to {lmax} with memory {memory_limit}"
            )

        if lmin == chain.ln:
            seq.insert(SeqLoss())
            return seq

        w = what[cmem][lmin][lmax]
        #  -- Solution 1 --
        if w[0]:
            k = w[1]
            sol = chain.body[lmin].sols[k]
            seq.insert(SeqBlockFe(lmin, sol.fwd_sched))
            seq.insert_seq(
                seq_builder_rec(lmin + 1, lmax, cmem - chain.cbw[lmin + 1][k])
            )
            seq.insert(SeqBlockBwd(lmin, sol.bwd_sched))

        #  -- Solution 1 --
        else:
            j = w[1]
            seq.insert(SeqBlockFc(lmin, chain.body[lmin].Fc_sched))
            for k in range(lmin + 1, j):
                seq.insert(SeqBlockFn(k, chain.body[k].Fn_sched))
            seq.insert_seq(seq_builder_rec(j, lmax, cmem - chain.cw[j]))
            seq.insert_seq(seq_builder_rec(lmin, j - 1, cmem))
        return seq

    #  ~~~~~~~~~~~~~~~~~~

    seq = seq_builder_rec(0, chain.ln, mmax)
    return seq


# ===================================
# =====  interface to C version =====
# ===================================

# The C version produces 'csequence' SeqOps, we have to convert them
import rockmate.csequence as cs

try:
    import rockmate.csolver as rs
    csolver_present = True
except:
    csolver_present = False


def convert_sequence_from_C(chain: RK_Chain, original_sequence):
    def convert_op(op):
        if isinstance(op, cs.SeqLoss):
            return SeqLoss()
        body = chain.body[op.index]
        if isinstance(op, cs.SeqBlockFn):
            return SeqBlockFn(op.index, body.Fn_sched)
        if isinstance(op, cs.SeqBlockFc):
            return SeqBlockFc(op.index, body.Fc_sched)
        if isinstance(op, cs.SeqBlockFe):
            return SeqBlockFe(op.index, body.sols[op.option].fwd_sched)
        if isinstance(op, cs.SeqBlockBwd):
            return SeqBlockBwd(op.index, body.sols[op.option].bwd_sched)

    result = RK_Sequence([convert_op(op) for op in original_sequence])
    return result


def csolve_dp_functionnal(chain: RK_Chain, mmax, opt_table=None):
    if opt_table is None:  ## TODO? if opt_table.mmax < mmax, create new table
        result = rs.RkTable(chain, mmax)
    else:
        result = opt_table
    result.get_opt(mmax)
    return result


def cseq_builder(chain: RK_Chain, mmax, opt_table):
    result = opt_table.build_sequence(mmax)
    return convert_sequence_from_C(chain, result)


# ===============================================
# =====  generic interface, selects version =====
# ===============================================


def solve_dp_functionnal(chain, mmax, opt_table=None, force_python=False):
    if force_python or not csolver_present:
        return psolve_dp_functionnal(chain, mmax, opt_table)
    else:
        return csolve_dp_functionnal(chain, int(mmax), opt_table)


def seq_builder(chain, mmax, opt_table):
    if csolver_present and isinstance(opt_table, rs.RkTable):
        return cseq_builder(chain, int(mmax), opt_table)
    else:
        return pseq_builder(chain, mmax, opt_table)

