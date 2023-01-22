#!/usr/bin/python

from . import parameters
from .sequence import *
from .griewank import convert_griewank_to_rotor, convert_griewank_to_rotor_xbar
from .utils import argmin
from itertools import accumulate

try:
    import dynamic_programs as dp
    c_version_present = True
except ImportError:
    c_version_present = False


def compute_table(chain, mmax):
    """ Return the Opt_hete tables
        for every Opt_hete[m][lmin][lmax-lmin] with lmin = 0...chain.length
        and lmax = lmin...chain.length (lmax is not included) and m = 0...mmax
        The computation uses a dynamic program"""
    fw = chain.fweigth
    bw = chain.bweigth
    cw = chain.cweigth
    cbw = chain.cbweigth + [0]
    # Build table
    opt = [[[] for _ in range(chain.length+1)] for _ in range(mmax + 1)]
    # Initialize borders of the tables for lmax-lmin = 0 and 1

    maxcw = { (i,j): max(cw[i:j]) for i in range(chain.length + 1) for j in range(i+1, chain.length + 2)}
    sumfw = list(accumulate(fw))
    sumbw = list(accumulate(bw))

    for m in range(mmax + 1):
        for i in range(chain.length+1):
            #lmax-lmin = 0
            if m >= max(cbw[i], cw[i] + cbw[i+1]):
                opt[m][i].append(bw[i])
            else:
                opt[m][i].append(float("inf"))
            #lmax - lmin = 1
            if i == chain.length:
                continue
            if m >= max(cbw[i], cw[i] + max(cbw[i+1], cbw[i+2] + max(cw[i], cw[i+1]))):
                opt[m][i].append(fw[i] + bw[i] + bw[i+1])
            else:
                opt[m][i].append(float("inf"))

    # Compute everything
    for m in range(mmax + 1):
        for i in range(chain.length+1):
            for l in range(2, chain.length -i + 1):
                mmin = max(cbw[i], cbw[i+1] + cw[i])
                mmin = max(mmin, max(cw[i] + cbw[j] + maxcw[(i, j)] for j in range(i+2,i+l+2)))
                if m < mmin:
                    opt[m][i].append(float("inf"))
                else:
                    # no_checkpoint = sum(bw[i:i+l+1]) + sum([sum(fw[i:i+k+1]) for k in range(l)])
                    no_checkpoint = sumbw[i+l] - (sumbw[i-1] if i > 0 else 0) + sum(sumfw[i+k] - (sumfw[i-1] if i > 0 else 0) for k in range(l))
                    if m < cw[i]:
                        value = float("inf")
                    else: 
                        try: 
                            # value = min([sum(fw[i:j]) + opt[m-cw[i]][j][i+l-j] + opt[m][i][j-i-1] for j in range(i+1, i+l)])
                            value = min(sumfw[j-1] - (sumfw[i-1] if i > 0 else 0) + opt[m-cw[i]][j][i+l-j] + opt[m][i][j-i-1] for j in range(i+1, i+l))
                        except IndexError as e: 
                            print("Error when computing for m={m} i={i} l={l} cw={c}".format(m=m, i=i, l=l, c=cw[i]))
                            raise e
                    opt[m][i].append(min(value, no_checkpoint))
    return opt


def griewank_heterogeneous_rec(chain, lmin, lmax, cmem, opt_table):
    """ chain : the class describing the AC graph
        lmin : index of the first forward to execute
        lmax : upper bound index of the last forward to execute (not included)
        cmem : number of available memory slots
        Return the optimal sequence of makespan Opt_hete[cmem][lmin][lmax-lmin]"""
    sequence = Sequence(Function("Griewank Heterogeneous", lmin, lmax, cmem))
    sequence.isStraightforward = True
    if opt_table[cmem][lmin][lmax-lmin] == float("inf"):
        raise ValueError("Can not process this chain from index {lmin} to {lmax} with memory {cmem}".format(lmin=lmin, lmax=lmax, cmem=cmem))
    if lmin == lmax:
        sequence.insert(Backward(lmin))
        return sequence
    if lmin +1 == lmax:
        sequence.insert(WriteMemory(lmin))
        sequence.insert(Forward(lmin))
        sequence.insert(Backward(lmin+1))
        sequence.insert(ReadMemory(lmin))
        sequence.insert(Backward(lmin))
        return sequence
    list_mem = [sum(chain.fweigth[lmin:j]) + opt_table[cmem-chain.cweigth[lmin]][j][lmax-j] + opt_table[cmem][lmin][j-lmin-1] for j in range(lmin+1, lmax+1)]
    no_checkpoint = sum(chain.bweigth[lmin:lmax+1]) + sum([sum(chain.fweigth[lmin:j]) for j in range(lmin+1,lmax+1)])
    if no_checkpoint <= min(list_mem):
        sequence.insert(WriteMemory(lmin))
        for j in range(lmax-1, lmin -1, -1):
            if j != lmax - 1:
                sequence.insert(ReadMemory(lmin))
            sequence.insert(Forwards(lmin, j))
            sequence.insert(Backward(j + 1))
        sequence.insert(ReadMemory(lmin))
        sequence.insert(Backward(lmin))
    else:
        jmin = lmin + argmin(list_mem) + 1
        sequence.insert(WriteMemory(lmin))
        sequence.insert(Forwards(lmin, jmin-1))
        sequence.insert_sequence(griewank_heterogeneous_rec(chain, jmin, lmax, cmem - chain.cweigth[lmin], opt_table))
        sequence.insert(ReadMemory(lmin))
        sequence.insert_sequence(griewank_heterogeneous_rec(chain, lmin, jmin-1, cmem, opt_table).remove_useless_write())
    return sequence


def griewank_heterogeneous(chain, memory_limit, useXbar = False, showInputs = False, force_python = False):
    max_peak = max(max(chain.fwd_tmp), max(chain.bwd_tmp))
    available_mem = memory_limit - max_peak

    if available_mem <= 0: 
        raise ValueError("Can not execute: memory {p} smaller than maximum peak {max_peak}".format(p=memory_limit, max_peak = max_peak))

    length = chain.length
    if useXbar:
        sizes = chain.cbweigth
    else: 
        sizes = chain.cweigth


    converted_chain = parameters.Chain(chain.fweigth, chain.bweigth, sizes, sizes, [0] * length, [0] * (length+1))
    ### ^ In Griewank Heterogeneous, cbw is the size of y_i, it is the same as x_i

    if showInputs: 
        print("Modified Inputs: %d -s '%s'" % (available_mem, converted_chain), file=sys.stderr)

    if c_version_present and not force_python:
        opt_hete = dp.griewank_heterogeneous_compute_table(converted_chain, available_mem)
    else: 
        opt_hete = compute_table(converted_chain, available_mem)
        
    seq = griewank_heterogeneous_rec(converted_chain, 0, length, available_mem, opt_hete)
    if useXbar:
        converted = convert_griewank_to_rotor_xbar(seq, length)
    else: 
        converted = convert_griewank_to_rotor(seq, length)
    return converted


if __name__ == '__main__':
    from .utils import simulate_sequence

    params = parameters.parse_arguments()
    start_time = time.time()
    seq = griewank_heterogeneous_rec(params.chain, 0, params.chain.length, params.cm, params)
    end_time = time.time()
    converted = convert_griewank_to_rotor_xbar(seq, params)
    print(seq)
    print(converted)
    maxUsage = simulate_sequence(converted, params.chain.length, chain=params.chain, display=False)
    print("Chain length: ", params.chain.length)
    ## print("Sequence: ", seq)
    print("Memory: ", maxUsage)
    print("Makespan: ", converted.makespan)
    print("Compute time: %.3f ms"%(1000*(end_time - start_time)))
