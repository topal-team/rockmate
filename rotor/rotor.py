import torch
import warnings

from . import memory
from . import inspection
from . import algorithms as alg
from .algorithms.sequence import *
from .utils import *

import sys
import math


class TensorStorage:
    def __init__(self):
        self.storage = {}  ## storage[i] stores the input of functions[i]
        self.sourceStorage = {}
        ## if storage[i] has a computation graph,
        ## sourceStorage[i] is the input to the
        ## call that created it
        self.rngStorage = {}

    def addValue(self, index, val, source, rng_state = None):
        self.storage[index] = val
        self.sourceStorage[index] = source
        self.rngStorage[index] = rng_state

    def getValue(self, index):
        return self.storage[index]
    def getSource(self, index):
        return self.sourceStorage[index]
    def getRng(self, index):
        return self.rngStorage[index]

    def deleteIndex(self, index):
        del self.storage[index]
        del self.sourceStorage[index]
        del self.rngStorage[index]

    def __str__(self, *args):
        def keyToStr(key):
            suffix = ""
            if self.sourceStorage[key] is not None: suffix += "*"
            if self.rngStorage[key] is not None: suffix += "^"
            return str(key)+suffix
        keyList = " ".join(keyToStr(k) for k in self.storage.keys())
        return "Size {}, keys {}".format(len(self.storage), keyList)

    # These methods are useful to be able to use save_for_backward()
    # for all the tensors that are kept between the forward and backward
    # phases. For the moment it is a noop (see commented sections below) 
    # because it does not work properly for our case. The missing feature
    # is the possibility to perform 
    # 
    # y = f(x)
    # save_for_backward(x, y)
    # 
    # in the forward phase, and then 
    # 
    # x, y = saved_tensors
    # y.backward(grad)
    # grad = x.grad
    #
    # in the backward phase. As of now, this results in `x.grad` being
    # `None` instead of containing useful data (see $580)
    # 
    # If this feature can not be implemented, this restricts the set of possible
    # checkpointing stategies. It is actually still possible to compute the
    # optimal sequence in this restricted set, so we would (only) have to adapt
    # our algorithms. 

    def serialize(self):
        self.result = tuple()
        def save(tensors):
            if tensors is None: return None
            tensors = ensure_tuple(tensors)
            startIndex = len(self.result)
            self.result = self.result + tensors
            endIndex = len(self.result)
            return (startIndex, endIndex)

        for k in self.storage.keys():
            pass
            # self.storage[k] = save(self.storage[k])
            # self.sourceStorage[k] = save(self.sourceStorage[k])
        result = self.result
        del self.result
        return result

    def _deserialize_helper(self, dictionary, saved_tensors):
        for k in dictionary.keys():
            if dictionary[k] is not None:
                start, end = dictionary[k]
                if end == start + 1:
                    dictionary[k] = saved_tensors[start]
                else:
                    dictionary[k] = saved_tensors[start:end]


    def deserialize(self, saved_tensors):
        pass 
        # self._deserialize_helper(self.storage, saved_tensors)
        # self._deserialize_helper(self.sourceStorage, saved_tensors)


class CheckpointOptim(torch.autograd.Function):
    r"""This computes a sequence of functions, following the sequence of
    operations given as argument. A selected subset of activations are
    stored during the forward phase, some with their computation
    graph, some without. The backward phase follows the end of the
    sequence, with some recomputation when values are missing.
    """

    
    @staticmethod
    def forward(ctx, functions, sequence, names, preserve_rng_state, arg):
        check_backward_validity(arg)
        input = arg
        ctx.run_function = functions
        ctx.names = names
        ctx.preserve_rng_state = preserve_rng_state
        storage = TensorStorage()
        storage.addValue(0, arg, None)
        sourceOfCurrent = None
        for idx, op in enumerate(sequence):
            if names: print(op, names[op.index] if hasattr(op, 'index') else "", file=sys.stderr)

            if type(op) is ForwardEnable:
                # A theorem says: ForwardEnable operations are never done twice. So there is no
                # need to save the RNG state here.
                storage.addValue(op.index, input, sourceOfCurrent)
                input = detach_variable(input, True)
                sourceOfCurrent = input
                with torch.enable_grad():
                    input = functions[op.index](input)

            elif isinstance(op, Forward): # covers both ForwardNograd and ForwardCheck
                if type(op) is ForwardCheck:
                    storage.addValue(op.index, input, sourceOfCurrent, RngState(input) if preserve_rng_state else None)
                with torch.no_grad():
                    input = functions[op.index](input)
                sourceOfCurrent = None
                
            elif type(op) is Loss:
                lossOperationIndex = idx
                break
            elif type(op) is Backward:
                raise ValueError("Encountered Backward op {op} in Forward phase, index {idx}".format(op=op, idx=idx))
            else:
                raise AttributeError("Unknown operation type {t} {op}".format(t=type(op), op=op))

        # Save the last computed value (it is always a ForwardEnable operation)
        if lossOperationIndex > 0: 
            lastIndex = sequence[lossOperationIndex-1].index
            storage.addValue(lastIndex + 1, input, sourceOfCurrent)
        ctx.sequence = sequence[lossOperationIndex + 1:]
        ctx.save_for_backward(*storage.serialize())
        ctx.storage = storage
        if lossOperationIndex > 0: 
            return detach_variable(input)
        else: 
            return input



    @staticmethod
    def backward(ctx, *args):
        names = ctx.names
        preserve_rng_state = ctx.preserve_rng_state
        sequence = ctx.sequence
        functions = ctx.run_function
        storage = ctx.storage
        storage.deserialize(ctx.saved_tensors)

        idx = 0
        while idx < len(sequence):
            op = sequence[idx]
            if names: print(op, names[op.index] if hasattr(op, 'index') else "", "Usage: ", storage, file=sys.stderr)
            if isinstance(op, Forward):
                input = storage.getValue(op.index)
                state = storage.getRng(op.index)
                source = None
                if type(op) is ForwardEnable:
                    input = detach_variable(input, True)
                    source = input
                    if state: storage.rngStorage[op.index] = None # no longer needed, we will not do this forward again

                if state: state.restore()
                elif type(op) is ForwardCheck and preserve_rng_state:
                    state = RngState(input)
                    storage.rngStorage[op.index] = state
                with torch.set_grad_enabled(type(op) is ForwardEnable):
                    input = functions[op.index](input)
                storage.addValue(op.index+1, input, source) # not saving state now, state will be saved if needed just before next Fwd
                if type(op) is ForwardNograd:
                    storage.deleteIndex(op.index)
                del input

            elif type(op) is Loss:
                raise ValueError("Encountered Loss op {op} in Backward phase, index {idx}".format(op=op, idx=idx))

            elif type(op) is Backward:
                src_index = op.index + 1
                torch.autograd.backward(storage.getValue(src_index), grad_tensors=args)
                args = get_gradients(storage.getSource(src_index))
                assert args is not None
                storage.deleteIndex(src_index)
                
            idx += 1

        if isinstance(args, torch.Tensor): 
            return (None, None, None, None, args)
        else: 
            return (None, None, None, None, *args)
            



class Checkpointable(torch.nn.Module):

    def __init__(self, model, input = None, mem_limit = None,
                 mem_slots = 500, verbosity = 0, force_python = False, preserve_rng_state = True):
        super(Checkpointable, self).__init__()
        self.model = model
        self.modules_and_names = inspection.extract_children_from_sequential(model)
        self.names, self.functions = (list(vals) for vals in zip(*self.modules_and_names))
        self.verbosity = verbosity
        self.mem_slots = mem_slots
        self.force_python = force_python
        self.preserve_rng_state = preserve_rng_state
        self.all_values = None
        self.chain = None
        self.sequence = None
        self.loss_tmp_memory_usage = 0
        self.mem_limit = mem_limit
        
        if input is not None:
            self.measure(input)
            if mem_limit is not None:
                self.compute_sequence(mem_limit)
        
    def measure(self, input):
        self.all_values = inspection.measure_everything(self.modules_and_names, input)
        self.params = None   ## Forget old params, they are out of date now.
        self.sequence = None
        
        if self.verbosity > 0:
            longest_name = max(self.names, key=len)
            fmt_string_h = "{:<%d} {:>7} {:>7} {:>11} {:>11} {:>11} {:>11}" % len(longest_name)
            fmt_string = "{:<%d} {:>7.2f} {:>7.2f} {:>11} {:>11} {:>11} {:>11}" % len(longest_name)
            print(fmt_string_h.format("Name", "Tf", "Tb", "xbar", "x", "tmpF", "tmpB"), file=sys.stderr)
            for val in zip(self.names, *self.all_values):
                print(fmt_string.format(*val[0:3], *map(memory.MemSize, val[3:])), file=sys.stderr)
            
            self.compute_min_sequence()
            mkspan_min_sequence = self.get_expected_makespan()
            memory_min_sequence = self.get_expected_memory()
            self.compute_pytorch_sequence()
            mkspan_py_sequence = self.get_expected_makespan()
            memory_py_sequence = self.get_expected_memory()
            print("Min. memory usage:", memory.MemSize(memory_min_sequence))
            print("Max. memory usage:", memory.MemSize(memory_py_sequence), " makespan ", mkspan_py_sequence)
            self.sequence = None

    def discretize(self, values): 
        return [ math.ceil(value / self.mem_unit) for value in values ]
                
    def makeParams(self, mem_limit):
        if self.all_values is None: 
            raise(ValueError("Checkpointable: measure() should be called before compute_sequence()"))
        fwd_time, bwd_time, xbar_sizes, x_sizes, fwd_tmp, bwd_tmp = self.all_values
            
        if mem_limit is not None:
            
            self.mem_unit = mem_limit // self.mem_slots
            xbar_sizes =  self.discretize(xbar_sizes)
            x_sizes    =  self.discretize(x_sizes) 
            fwd_tmp = self.discretize(fwd_tmp)
            bwd_tmp = self.discretize(bwd_tmp + [self.loss_tmp_memory_usage])
            mem_slots = self.mem_slots
            
            if self.verbosity > 1: print('Opt Checkpoint: length = {}, memory = {}, unit = {}, slots = {}, sum xb = {}'
                                    ''.format(len(self.functions), memory.MemSize(mem_limit), memory.MemSize(self.mem_unit), self.mem_slots, sum(xbar_sizes)), file=sys.stderr)
        else:
            bwd_tmp = bwd_tmp + [self.loss_tmp_memory_usage]
            mem_slots = None
            self.mem_unit = 1
            
        self.chain = alg.Chain(fwd_time, bwd_time + [0], x_sizes, xbar_sizes, fwd_tmp, bwd_tmp)

    def check_sequence(self):
        if self.sequence is None: 
            raise(ValueError("Checkpointable: compute_sequence() should be called before forward()"))

        
    def compute_sequence(self, mem_limit=None, mem_slots = None, force_python = None, floating = False):
        device = next(self.model.parameters()).device
        if mem_limit is None:
            mem_limit = int(memory.MeasureMemory(device).available() * 0.9)
        # Check that we can actually book this much memory
        # torch.cuda.empty_cache()
        # tmp = torch.zeros(int(mem_limit/4), device=device)
        # del tmp
        if mem_slots: self.mem_slots = mem_slots
        if force_python is not None: self.force_python = force_python
        self.makeParams(mem_limit)
        if self.verbosity > 2: print("Inputs: %d -s '%s'" % (self.mem_slots, self.chain), file=sys.stderr)

        if floating:
            self.sequence = alg.floating(self.chain, self.mem_slots, force_python = force_python)
        else: 
            self.sequence = alg.persistent(self.chain, self.mem_slots, force_python = force_python)

    def get_expected_memory(self):
        self.check_sequence()
        exp_memory = alg.simulate_sequence(self.sequence, None, chain=self.chain, display = False)
        return exp_memory * self.mem_unit

    def get_memory_used_during_loss(self): 
        self.check_sequence()
        exp_memory = alg.simulate_sequence(self.sequence, None, chain=self.chain, display = False, stopAtLoss = True)
        return exp_memory * self.mem_unit
    
    def get_expected_makespan(self):
        self.check_sequence()
        return self.sequence.get_makespan(self.chain)

    def compute_seq_sequence(self, segments = None):
        self.makeParams(None)
        self.sequence = alg.chen_sqrt(self.chain.length, segments)

    def compute_pytorch_sequence(self):
        self.makeParams(None)
        self.sequence = alg.no_checkpoint(self.chain.length)    

    def compute_min_sequence(self): 
        self.makeParams(None)
        self.sequence = alg.recompute_all(self.chain.length)    
        

    def compute_homogeneous_sequence(self, mem_limit, mem_slots = None, useXbar = False):
        if mem_slots: self.mem_slots = mem_slots
        self.makeParams(mem_limit)
        self.sequence = alg.griewank(self.chain, self.mem_slots, useXbar, showInputs = self.verbosity > 2)

    def compute_heterogeneous_sequence(self, mem_limit, mem_slots = None, useXbar = False):
        if mem_slots: self.mem_slots = mem_slots
        self.makeParams(mem_limit)
        self.sequence = alg.griewank_heterogeneous(self.chain, self.mem_slots, useXbar,
                                                   showInputs = self.verbosity > 2)
        
        
        
    def display(self):
        self.check_sequence()
        exp_memory = alg.simulate_sequence(self.sequence, None, chain=self.chain, display = self.verbosity > 5)
        if self.verbosity > 0:
            if self.verbosity > 1: print("Actions:", repr(self.sequence), file=sys.stderr)
            print("Expected makespan:", self.get_expected_makespan(),
                  "memory: %d/%d, %s (%d)" % (exp_memory, self.mem_slots, memory.MemSize(exp_memory * self.mem_unit), exp_memory * self.mem_unit), file=sys.stderr)
        

    def forward(self, inputs):
        if self.training: 
            if self.all_values is None:
                self.measure(inputs)
            if self.sequence is None:
                self.compute_sequence(self.mem_limit)
            if self.verbosity > 0: 
                self.display()
            strippedSequence, startOfSuffix = self.sequence.withoutSuffix()
            if self.verbosity > 1: print("Stripped sequence:", strippedSequence, file=sys.stderr)
            inputs = CheckpointOptim.apply(self.functions, strippedSequence.list_operations(), self.names if self.verbosity > 3 else None, self.preserve_rng_state, inputs)
            if startOfSuffix is not None: 
                for i in range(startOfSuffix, len(self.functions)):
                    inputs = self.functions[i](inputs)
            return inputs
        else:
            return self.model(inputs)


