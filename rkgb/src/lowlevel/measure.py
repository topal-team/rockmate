
import torch
import numpy as np
import time
import os
import psutil
import subprocess

# Simplified version of Rotor measure

def tensor_memory_size(t):
    if isinstance(t, torch.Tensor):
        return t.element_size() * np.prod(t.shape)
    else:
        return sum(tensor_memory_size(u) for u in t)

def pretty_format_memory(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

# =========
# = timer =
# =========

class Timer:
    # To get robust results:
    minimum_total_duration = 0
    minimum_number_of_iterations = 5
    
    def measure(self, func, iterations = 1):
        self.start()
        for _ in range(iterations):
            func()
        self.end()
        return self.elapsed() / iterations
    
    def robust_measure(self, func, reset_func = None):
        """
        Measures the time it takes for `func` to be executed.
        Do several iterations to get a robust result,
        the final result is the average of all the iterations,
        excluding the two extreme values:
        `result = [sum(measures)-max(measures)-min(measures)] / (len-2)`

        `reset_func` is executed between each call to `func`,
        to reset the execution environment.
        """
        elapsed_time = self.measure(func)
        nb_iterations = 1
        measures = [elapsed_time]
        total_elapsed_time = elapsed_time
        while (total_elapsed_time < self.minimum_total_duration
        or nb_iterations < self.minimum_number_of_iterations):
            if reset_func is not None:
                reset_func()
            elapsed_time = self.measure(func)
            measures.append(elapsed_time)
            total_elapsed_time += elapsed_time
            nb_iterations += 1
        if nb_iterations > 2:
            return (
                (sum(measures)-max(measures)-min(measures))
                /(len(measures)-2))
        else:
            return np.median(measures)
        
    # TO OVERWRITE:
    def start(self): pass
    def end(self): pass
    def elapsed(self): pass


class TimerCPU(Timer):
    def reset(self):
        self.elapsed_time = None
    def start(self):
        self.elapsed_time = time.perf_counter()
    def end(self):
        self.elapsed_time = time.perf_counter() - self.elapsed_time
    # In milliseconds
    def elapsed(self):
        return self.elapsed_time * 1000


class TimerCUDA(Timer):
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

# =========



# =================
# = MeasureMemory =
# =================

class MemoryTracker:
    def __init__(self,device):
        self.device = device
        self.last_memory = self.current()
    def diff_compared_to_last(self):
        current = self.current()
        result = current - self.last_memory
        self.last_memory = current
        return result
    def measure(self, func, *args):
        self.last_memory = self.current()
        self.reset_max()
        max_before = self.maximum()
        result = func(*args) # run
        usage = self.diff_compared_to_last()
        max_usage = self.maximum() - max_before
        return result, usage, max_usage
    
    # TO OVERWRITE:
    def maximum(self): pass
    def current(self): pass
    def reset_max(self): pass


class MemoryTrackerCPU(MemoryTracker):
    """
    TO DO : Right now it's not usable, as we miss `maximum_value`
    Would be nice to add, but not urgent
    Could use:
    https://docs.python.org/3/library/tracemalloc.html
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.max_memory = 0
        super().__init__(torch.device("cpu"))
    def maximum(self):
        return self.max_memory
    def current(self):
        result = int(self.process.memory_info().rss)
        self.max_memory = max(self.max_memory, result)
        return result
    def reset_max(self):
        self.max_memory = 0
        self.max_memory = self.current()
    
    
class MemoryTrackerCUDA(MemoryTracker):
    def maximum(self):
        return torch.cuda.max_memory_allocated(self.device)
    def current(self):
        return torch.cuda.memory_allocated(self.device)
    def reset_max(self):
        return torch.cuda.reset_max_memory_allocated(self.device)
    
    # Optional / to help debug
    def available(self, index=None):
        result = subprocess.check_output(
            ["nvidia-smi", 
             "--query-gpu=memory.free", 
             "--format=csv,nounits,noheader"])
        l = [int(x) for x in result.strip().split(b"\n")]
        if index is None:
            index = self.device.index
        if index is None: index = torch.cuda.current_device()
        return (
            l[index]*1024*1024 
            + torch.cuda.memory_cached(self.device) 
            - torch.cuda.memory_allocated(self.device))

# =================