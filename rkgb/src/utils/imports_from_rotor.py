# ==========================
# =  REIMPLEMENTATION OF   =
# =  ROTOR FUNCTIONS USED  =
# ==========================

import torch
import numpy as np
import time
import os
import psutil
import subprocess
# import statistics # unused

# -> We don't want to include Rotor repository
# -> in Rockmate since we use very few of their
# -> functions. But the following functions 
# -> belong to Rotor authors.

# =================
# = class MemSize =
# =================

def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

class MemSize:
    def __init__(self, v):
        self.v = v

    def __add__(self, other):
        return self.__class__(self.v + other.v)
    def __sub__(self, other):
        return self.__class__(self.v - other.v)
    def __neg__(self): # new
        return self.__class__(-self.v)
    
    @classmethod
    def fromStr(cls, str):
        suffixes = {'k': 1024, 'M': 1024*1024, 'G': 1024*1024*1024}
        if str[-1] in suffixes:
            val = int(float(str[:-1]) * suffixes[str[-1]])
        else:
            val = int(str)
        return MemSize(val)

    def __str__(self):
        return sizeof_fmt(self.v)

    def __format__(self, fmt_spec):
        return sizeof_fmt(self.v).__format__(fmt_spec)
    
    def __repr__(self):
        return str(self.v)

    def __int__(self):
        return self.v

    def __eq__(self,m):
        return self.v == m.v
    def __hash__(self):
        return id(self)

# =================



# ===============
# = tensorMsize =
# ===============

def tensorMsize(t):
    if isinstance(t, torch.Tensor):
        return t.element_size() * np.prod(t.shape)
    else:
        return sum(tensorMsize(u) for u in t)

# ===============



# =========
# = timer =
# =========

class Timer:
    def measure(self, func, iterations = 1):
        self.start()
        for _ in range(iterations):
            func()
        self.end()
        return self.elapsed() / iterations

    """ # -> Reimplemented in def_inspection
    def measure_median(self, func, samples = 3, **kwargs):
        values = []
        for _ in range(samples):
            values.append(self.measure(func, **kwargs))
        return statistics.median(values)
    """
            
class TimerSys(Timer):
    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed_time = None
    
    def start(self):
        self.elapsed_time = time.perf_counter()

    def end(self):
        self.elapsed_time = time.perf_counter() - self.elapsed_time

    # In milliseconds
    def elapsed(self):
        return self.elapsed_time * 1000

    def elapsedAndReset(self):
        self.end()
        result = self.elapsed()
        self.reset()
        self.start()
        return result

class TimerCuda(Timer):
    def __init__(self, device):
        self.device = device
        self.stream = torch.cuda.current_stream(device)
        self.reset()

    def reset(self):
        self.startEvent = torch.cuda.Event(enable_timing = True)
        self.endEvent = torch.cuda.Event(enable_timing = True)

    def start(self):
        self.startEvent.record(self.stream)

    def end(self):
        self.endEvent.record(self.stream)
        torch.cuda.synchronize(self.device)

    # In milliseconds
    def elapsed(self):
        return self.startEvent.elapsed_time(self.endEvent)

    def elapsedAndReset(self):
        self.end()
        result = self.elapsed()
        self.reset()
        self.start()
        return result
    
def make_timer(device):
    if device.type == 'cuda':
        return TimerCuda(device)
    else:
        return TimerSys()

# =========



# =================
# = MeasureMemory =
# =================

class MeasureMemory:
    def __init__(self, device):
        self.device = device
        self.cuda = self.device.type == 'cuda'
        if not self.cuda:
            self.process = psutil.Process(os.getpid())
            self.max_memory = 0
        self.last_memory = self.currentValue()
        self.start_memory = self.last_memory

    def currentValue(self):
        if self.cuda:
            result = torch.cuda.memory_allocated(self.device)
        else: 
            result = int(self.process.memory_info().rss)
            self.max_memory = max(self.max_memory, result)
        return result
        
    def maximumValue(self):
        if self.cuda:
            return MemSize(torch.cuda.max_memory_allocated(self.device))
        else:
            return MemSize(self.max_memory)

    def available(self, index=None):
        assert self.cuda
        result = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"])
        l = [int(x) for x in result.strip().split(b"\n")]
        if index is None:
            index = self.device.index
        if index is None: index = torch.cuda.current_device()
        return l[index]*1024*1024 + torch.cuda.memory_cached(self.device) - torch.cuda.memory_allocated(self.device)
        
    ## Requires Pytorch >= 1.1.0
    def resetMax(self):
        if self.cuda:
            torch.cuda.reset_max_memory_allocated(self.device)
        else:
            self.max_memory = 0
            self.max_memory = self.currentValue()
        

    def current(self):
        return MemSize(self.currentValue())
        
    def diffFromLast(self):
        current = self.currentValue()
        result = current - self.last_memory
        self.last_memory = current
        return MemSize(result)

    def diffFromStart(self):
        current = self.currentValue()
        return MemSize(current - self.start_memory)

    def currentCached(self):
        if not self.cuda: 
            return 0
        else: 
            return MemSize(torch.cuda.memory_cached(self.device))

    def measure(self, func, *args):
        self.diffFromLast()
        self.resetMax()
        maxBefore = self.maximumValue()
        result = func(*args)
        usage = self.diffFromLast()
        maxUsage = self.maximumValue() - maxBefore

        return result, usage.v, maxUsage.v

# =================