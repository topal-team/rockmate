# rotor: Rematerializing Optimally with pyTORch

## Description

**Purpose:** This code is meant to replace torch/utils/checkpoint.py,
by providing more efficient checkpointing strategies. The algorithm is
easier to tune, as the required parameter as input is the available
memory instead of the number of segments.

For more details about the algorithm inside `rotor`, and for
experimental validation, see our paper at
https://hal.inria.fr/hal-02352969.

## Installation

Beside the obvious requirement on `pytorch`, `rotor` has only one
prerequisite: `psutil`. For installation, just run

```
python setup.py install
```

This will also compile the C extension which contains faster
implementations of the dynamic programs. 

## Usage

### Standard usage
The main class is `rotor.Checkpointable`. To use it:
* Describe your module as a `torch.Sequential` module
* Create a sequential module from it:
  ```
  chk = Checkpointable(module)
  ```
* Provide a sample input to perform timing and memory
  measurements of your module
  ```
  input = torch.rand(shape)
  chk.measure(input)
  ```
* Compute the optimal sequence for a given memory limit (in bytes)
  ```
  chk.compute_sequence(500*1024*1024)
  ```
* Use the Checkpointable as a module
  ```
  output = chk(input)
  output.backward()
  grad = input.grad
  ```

As an alternative, the sample input and memory limit can be specified
in the construction of `Checkpointable`:
```
input = torch.rand(shape)
chk = Checkpointable(module, input, 500*1024*1024)
output = chk(input)
```

Or only the sample input:
```
input = torch.rand(shape)
chk = Checkpointable(module, input)
chk.compute_sequence(500*1024*1024)
output = chk(input)
```

### Implemented models
Rotor also contains a suitable adaptation of the main models available
in `torchvision`: ResNet, VGG, Inception, and Densenet, available
respectively as `rotor.resnet`, `rotor.vgg`, `rotor.inception`, and
`rotor.densenet`. Each of these modules has the same interface as the
corresponding `torchvision` version.


### Complete example

Here is a complete working example, which runs ResNet18 with a batch
size of 32 with a memory usage of 700MB, whereas making the same run
without checkpointing would use 843MB:

```
import rotor
import torch

device = torch.device("cuda")
net = rotor.models.resnet18()
net.to(device=device)
net_check = rotor.Checkpointable(net)
shape = (32, 3, 224, 224)
sample = torch.rand(*shape, device=device)
net_check.measure(sample)
net_check.compute_sequence(mem_limit = 700*1024*1024)

data = torch.rand(*shape, device=device)
data.requires_grad = True
result = net_check(data).sum()
result.backward()
grad = data.grad
```

### Notes: 
* As of now, for technical reasons rotor does not use the
  `save_for_backward()` interface that detects when tensors are
  changed between the forward and the backward phase. Make sure to not
  modify the tensors.

* Just like `torch/utils/checkpoint.py`, `Checkpointable` preserves RNG
  states by default at each checkpoint, and restores it during the
  backward phase.
  
* Unclear to us how to portably get CPU resource usage information.
  Our code (`memory.py`) uses `import resource` to allow measuring
  resource usage even when not running on CUDA. If portability to
  non-Unix platforms is important, either this will require to
  remove this feature or use a more portable alternative.
