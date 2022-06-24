import torch
import warnings

def detach_variable(inputs, force_required_grad = False):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = force_required_grad or inp.requires_grad
            out.append(x)
        return tuple(out)
    elif isinstance(inputs, torch.Tensor):
        out = inputs.detach()
        out.requires_grad = force_required_grad or inputs.requires_grad
        return out
    else: 
        raise RuntimeError(
            "Only Tensor or tuple of Tensors is supported. Got Unsupported input type: ", type(inputs).__name__)
    
def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


def get_device(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.device
    else:
        result = None
        for inp in inputs:
            if  isinstance(inp, torch.Tensor):
                if result is None:
                    result = inp.device
                if result != inp.device:
                    raise ValueError("Two Tensors in the input have"
                                     " different devices {} and {}".format(result, inp.device))
        if result is None:
            raise ValueError("At least one input should be a Tensor")
        return result

def ensure_tuple(output):
    if isinstance(output, torch.Tensor):
        return (output,)
    return output

def get_gradients(inputs):
    if isinstance(inputs, torch.Tensor):
        return inputs.grad
    else:
        return tuple(inp.grad if isinstance(inp, torch.Tensor) else inp
                     for inp in inputs)

def remove_gradients(inputs):
    if isinstance(inputs, torch.Tensor):
        inputs.grad = None
    else:
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                inp.grad = None

class EmptyManager:
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_value, traceback):
        return False

class RngState:
    counter = 0

    def __init__(self, tensors):
        self.counter = RngState.counter
        RngState.counter += 1
        self.cpu_state = torch.get_rng_state()
        self.had_cuda = False
        if torch.cuda._initialized:
            self.had_cuda = True
            self.gpu_devices = list(set(t.get_device() for t in tensors
                                        if isinstance(t, torch.Tensor) and t.is_cuda))
            self.gpu_states = []
            for device in self.gpu_devices:
                with torch.cuda.device(device):
                    self.gpu_states.append(torch.cuda.get_rng_state())


    def restore(self):
        devices = self.gpu_devices if self.had_cuda else []
        torch.set_rng_state(self.cpu_state)
        if self.had_cuda:
            for device, state in zip(self.gpu_devices, self.gpu_states):
                with torch.cuda.device(device):
                    torch.cuda.set_rng_state(state)
