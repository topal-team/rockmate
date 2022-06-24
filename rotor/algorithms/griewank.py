#!/usr/bin/python

from . import parameters
from .sequence import *
from .utils import argmin
import argparse

def convert_griewank_to_rotor(seq, length):
    res = Sequence(Function("Conversion", seq.function.name, *seq.function.args))
    ops = seq.list_operations()

    idx = 0
    nb_ops = len(ops)
    pleaseSaveNextFwd = True
    while idx < nb_ops:
        op = ops[idx]
        # print(idx, op)
        if isForward(op):
            index = op.index
            if type(index) is int: index = (index, index)
            for i in range(index[0], index[1]+1):
                if pleaseSaveNextFwd:
                    res.insert(ForwardCheck(i))
                    pleaseSaveNextFwd = False
                else: res.insert(ForwardNograd(i))
        elif type(op) is WriteMemory:
            pleaseSaveNextFwd = True
        elif type(op) is DiscardMemory:
            pass
        elif type(op) is ReadMemory:
            if type(ops[idx+1]) is Backward:
                assert op.index == ops[idx+1].index
                res.insert(ForwardEnable(op.index))
                res.insert(Backward(op.index))
                idx += 1
            else:
                assert isForward(ops[idx+1])
                pleaseSaveNextFwd = True
        elif type(op) is Backward:
            if op.index == length:
                res.insert(Loss())
            else:
                res.insert(ForwardEnable(op.index))
                res.insert(Backward(op.index))
        else: 
            raise AttributeError("Unknown operation type {t} {op}".format(t=type(op), op=op))
        idx += 1

    return res

def convert_griewank_to_rotor_xbar(seq, length):
    res = Sequence(Function("Conversion Xbar", seq.function.name, *seq.function.args))
    ops = seq.list_operations()

    ## lastValueSaved = {i: None for i in range(0, length + 1)}
    idx = 0
    nb_ops = len(ops)
    pleaseSaveNextFwd = True
    nextFwdEnable = False
    doNotDoNextFwd = False
    while idx < nb_ops:
        op = ops[idx]
        # print(idx, op)
        if isForward(op):
            index = op.index
            if type(index) is int: index = (index, index)
            for i in range(index[0], index[1]+1):
                if doNotDoNextFwd:
                    doNotDoNextFwd = False
                elif nextFwdEnable: 
                    res.insert(ForwardEnable(i))
                    pleaseSaveNextFwd = True
                    nextFwdEnable = False
                elif pleaseSaveNextFwd:
                    res.insert(ForwardCheck(i))
                    pleaseSaveNextFwd = False
                else: res.insert(ForwardNograd(i))
            pleaseSaveNextFwd = False
        elif type(op) is WriteMemory:
            nextFwdEnable = True
        elif type(op) is DiscardMemory:
            pass
        elif type(op) is ReadMemory:
            if type(ops[idx+1]) is Backward:
                assert op.index == ops[idx+1].index
                res.insert(Backward(op.index))
                idx += 1
            else:
                assert isForward(ops[idx+1])
                doNotDoNextFwd = True
                pleaseSaveNextFwd = True
        elif type(op) is Backward:
            if op.index == length:
                res.insert(Loss())
            else:
                res.insert(ForwardEnable(op.index))
                res.insert(Backward(op.index))
        else: 
            raise AttributeError("Unknown operation {t} {op}".format(t=type(op), op=op))
        idx += 1

    return res


def get_opt_table(lmax, mmax):
    """ Return the Optimal table:
        every Opt[l][m] with l = 0...lmax and m = 0...mmax
        The computation uses a dynamic program"""
    uf = 1
    ub = 1
   
    # Build table
    ## print(mmax,lmax)
    opt = [[float("inf")] * (mmax + 1) for _ in range(lmax + 1)]
    # Initialize borders of the table
    for m in range(mmax + 1):
        opt[0][m] = ub
    for m in range(1, mmax + 1):
        opt[1][m] = uf + 2 * ub
    for l in range(1, lmax + 1):
        opt[l][1] = (l+1) * ub + l * (l + 1) / 2 * uf
    # Compute everything
    for m in range(2, mmax + 1):
        for l in range(2, lmax + 1):
            opt[l][m] = min(j * uf + opt[l - j][m-1] + opt[j-1][m] for j in range(1, l))
    return opt


def griewank_rec(l, cmem, opt_table):
    """ l : number of forward step to execute in the AC graph
        cmem : number of available memory slots
        Return the optimal sequence of makespan Opt_table(l, cmem)"""
    if cmem == 0:
        raise ValueError("Can not process a chain without memory")
    uf = 1
    sequence = Sequence(Function("Griewank", l, cmem))
    if l == 0:
        sequence.insert(Backward(0))
        sequence.insert(DiscardMemory(0))
        return sequence
    elif l == 1:
        sequence.insert(WriteMemory(0))
        sequence.insert(Forward(0))
        sequence.insert(Backward(1))
        sequence.insert(ReadMemory(0))
        sequence.insert(Backward(0))
        sequence.insert(DiscardMemory(0))
        return sequence
    elif cmem == 1:
        sequence.insert(WriteMemory(0))
        for index in range(l - 1, -1, -1):
            if index != l - 1:
                sequence.insert(ReadMemory(0))
            sequence.insert(Forwards(0,index))
            sequence.insert(Backward(index + 1))
        sequence.insert(ReadMemory(0))
        sequence.insert(Backward(0))
        sequence.insert(DiscardMemory(0))
        return sequence
    list_mem = [j * uf + opt_table[l-j][cmem-1] + opt_table[j-1][cmem] for j in range(1,l)]
    jmin = 1 + argmin(list_mem)
    sequence.insert(WriteMemory(0))
    sequence.insert(Forwards(0, jmin - 1))
    sequence.insert_sequence(griewank_rec(l - jmin, cmem - 1, opt_table).shift(jmin))
    sequence.insert(ReadMemory(0))
    sequence.insert_sequence(griewank_rec(jmin-1, cmem, opt_table).remove_useless_write())
    return sequence


def griewank(chain, memory_limit, useXbar = False, showInputs = False):
    max_peak = max(max(chain.fwd_tmp), max(chain.bwd_tmp))
    available_mem = memory_limit - max_peak
    if useXbar:
        sizes = sorted(chain.cbweigth, reverse = True)
    else: 
        sizes = sorted(chain.cweigth, reverse = True)
    nb_slots = 0
    sum = 0
    while sum < available_mem and nb_slots < len(sizes):
        sum += sizes[nb_slots]
        nb_slots += 1

    if showInputs: print("Hom Inputs: {l} {cm}".format(l=chain.length, cm=nb_slots), file=sys.stderr)

    opt_table = get_opt_table(chain.length, nb_slots)

    seq = griewank_rec(chain.length, nb_slots, opt_table)
    if useXbar:
        converted = convert_griewank_to_rotor_xbar(seq, chain.length)
    else: 
        converted = convert_griewank_to_rotor(seq, chain.length)
    return converted


