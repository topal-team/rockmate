import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class AtomicSequential(nn.Sequential):
    def __init__(self, *args, **kwargs):
        super(AtomicSequential, self).__init__(*args, **kwargs)
        self.not_really_sequential = True

## Because in-place operations do not mix well with partial checkpointing
## This class performs an in-place ReLU operation after
## the given module

class ReLUatEnd(nn.Module):
    def __init__(self, module):
        super(ReLUatEnd, self).__init__()
        self.module = module
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.module(x)
        x = self.relu(x)
        return x

# This is a particular version using BatchNorm2d because it is very often used
class BatchNorm2dAndReLU(nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rotor_relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = super().forward(x)
        x = self._rotor_relu(x)
        return x

# Because some models are slightly restructured compared to torchvision
# (for example with the ReLUatEnd class above),
# some keys of the state dict may need to be renamed to allow
# using pretrained models

def include_string_in_keys(string, prefixes, state_dict):
    for key in list(state_dict.keys()):
        for n in prefixes:
            if key.startswith(n):
                new_key = n + string + key[len(n):]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

    
class BothOutputs(nn.Module):

    def __init__(self, module_left, module_right, concat):
        super(BothOutputs, self).__init__()
        self.left = module_left
        self.right = module_right
        self.concat = concat

    def forward(self, x):
        hasRightValue = False
        if self.training and self.right: 
            rightValue = self.right(x)
            hasRightValue = True
        x = self.left(x)
        if hasRightValue:
            return self.concat(x, rightValue)
        return x

    
