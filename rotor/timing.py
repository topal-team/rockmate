import torch
import sys
import time
import statistics

class Timer:
    def measure(self, func, iterations = 1):
        self.start()
        for _ in range(iterations):
            func()
        self.end()
        return self.elapsed() / iterations

    def measure_median(self, func, samples = 3, **kwargs):
        values = []
        for _ in range(samples):
            values.append(self.measure(func, **kwargs))
        return statistics.median(values)
            
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

class ProgressTimer:
    def __init__(self, timer, displayFunc):
        self.timer = timer
        self.ongoingBwd = None
        self.timer.start()
        self.displayFunc = displayFunc
        self.additionalLength = 4
        
    def print(self, text):
        elapsed = self.timer.elapsedAndReset()
        self.displayFunc(text, time=elapsed)

    def endOngoing(self): 
        if self.ongoingBwd:
            self.print("Bwd " + self.ongoingBwd)
        
    def startBwd(self, name):
        self.endOngoing()
        self.ongoingBwd = name

    def startFwd(self, name):
        self.endOngoing()
        self.ongoingBwd = None

    def endFwd(self, name):
        assert(self.ongoingBwd is None)
        self.print("Fwd " + name)

    def endBwd(self, name):
        self.ongoingBwd = name
        self.endOngoing()
        self.ongoingBwd = None
