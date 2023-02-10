import rockmate_csolver as rs
import time
import pickle 


from rockmate import rotor_solver
from rockmate.csequence import (
    SeqBlockFn,
    SeqBlockFc,
    SeqBlockFe,
    SeqBlockBwd,
    SeqLoss,
    RK_Sequence,
)
from pgb.utils.global_vars import ref_verbose

class FakeChain:
    def __init__(self, d):
        self.__dict__.update(d)

def fake_seq_builder(chain, memory_limit, opt_table=({}, {})):
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
            seq.insert(SeqBlockFe(lmin, k))
            seq.insert_seq(
                seq_builder_rec(lmin + 1, lmax, cmem - chain.cbw[lmin + 1][k])
            )
            seq.insert(SeqBlockBwd(lmin, k))

        #  -- Solution 1 --
        else:
            j = w[1]
            seq.insert(SeqBlockFc(lmin))
            for k in range(lmin + 1, j):
                seq.insert(SeqBlockFn(k))
            seq.insert_seq(seq_builder_rec(j, lmax, cmem - chain.cw[j]))
            seq.insert_seq(seq_builder_rec(lmin, j - 1, cmem))
        return seq

    #  ~~~~~~~~~~~~~~~~~~

    seq = seq_builder_rec(0, chain.ln, mmax)
    return seq


def compare_ops(a, b):
    return str(a) == str(b) # not the nicest, but fastest to code ;-)

def compare_seqs(sa, sb):
    if (sa is None) and (sb is None):
        return True
    if (sa is None) or (sb is None):
        return False
    return all(compare_ops(a, b) for (a, b) in zip(sa.seq, sb.seq))


def one_test(ch, target):
    mmax = target - ch.cw[0]

    start = time.time()
    ## ref_verbose[0] = True
    opt_table = rotor_solver.solve_dp_functionnal(ch, mmax)
    ## ref_verbose[0] = False
    try:
        pyseq = fake_seq_builder(ch, target, opt_table)
    except ValueError:
        pyseq = None
    py_dur = time.time() - start

    start = time.time()
    result = rs.RkTable(ch, mmax)
    try:
        cseq = RK_Sequence(result.build_sequence(mmax))
    except ValueError:
        cseq = None
    c_dur = time.time() - start
    print(target, py_dur, c_dur, compare_seqs(pyseq, cseq))

def one_test_conly(ch, target):
    mmax = target - ch.cw[0]

    start = time.time()
    result = rs.RkTable(ch, mmax)
    try:
        cseq = RK_Sequence(result.build_sequence(mmax))
    except ValueError:
        cseq = None
    c_dur = time.time() - start
    print(target, c_dur)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("test_csolver")
    parser.add_argument("pkl_file", help="PKL file to open")
    parser.add_argument("--c-only", action="store_true", help="input is large, only solve with C")
    args = parser.parse_args()
    with open(args.pkl_file, "rb") as f:  ## "./test_csolver/example_DP_chain_large.pkl"
        inp = pickle.load(f)
    ch = FakeChain(inp)
    if args.c_only:
        for v in range(1000, 3000, 100):
            one_test_conly(ch, v)
    else:
        for v in range(500, 800, 50):
            one_test(ch, v)
