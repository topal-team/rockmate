import torch
import sys
import psutil
import os
import subprocess
from . import timing
from . import inspection


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

        return result, usage, maxUsage

class DisplayMemory:
    def __init__(self, device, maxLabelSize = 45):
        self.device = device
        self.memUsage = MeasureMemory(device)
        self.setMaxLabelSize(maxLabelSize)
        self.progress = None

    def setMaxLabelSize(self, size): 
        self.maxLabelSize = size
        self.formatStringTime = "{:<%d} {:>7.2f} TotalMem: {:>12} max reached: {:>12} wrt to last: {:>12} cached: {:>12}" % self.maxLabelSize
        self.formatStringNoTime = "{:<%d}         TotalMem: {:>12} max reached: {:>12} wrt to last: {:>12} cached: {:>12}" % self.maxLabelSize

        
    def printCurrentState(self, *args, **kwargs):
        if self.progress:
            self.progress.startFwd(None)
        self._printCurrentState(*args, **kwargs)

    def _printCurrentState(self, label, time=None):
        current = self.memUsage.current()
        maxUsed = self.maximumValue()
        fromLast = self.memUsage.diffFromLast()
        cached = self.memUsage.currentCached()
        if time: 
            print(self.formatStringTime.format(label, time, current, maxUsed, fromLast, cached))
        else: 
            print(self.formatStringNoTime.format(label, current, maxUsed, fromLast, cached))

    def maximumValue(self):
        return self.memUsage.maximumValue()

    def inspectModule(self, module):
        self.progress = timing.ProgressTimer(timing.make_timer(self.device), self._printCurrentState)
        maxLength = 0
        for (name, m) in inspection.extract_children_from_sequential(module):
            maxLength = max(maxLength, len(name))
            m.register_forward_hook(lambda x, y, z, n = name: self.progress.endFwd(n))
            m.register_forward_pre_hook(lambda x, y, n = name: self.progress.startFwd(n))
            m.register_backward_hook(lambda x, y, z, n = name: self.progress.endBwd(n))
        self.setMaxLabelSize(maxLength + self.progress.additionalLength)
        self.progress.startFwd(None)

        # ## For more inspection if desired
        # for (name, p) in module.named_parameters(): 
        #     p.register_hook(lambda g, m = name: self._printCurrentState("Param " + m))
