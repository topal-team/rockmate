# Rockmate

The `Rockmate` framework is designed for training a PyTorch neural network within a given GPU budget
constraint using automatic re-materialization (activation checkpointing) technique.

Given a PyTorch model, a sample input, and a GPU memory budget, `Rockmate` builds a new
`torch.nn.Module`, which performs forward and backward pass keeping activations under the given
budget.

- The new model produces the same outputs and gradients as the original one.
- Model training with a budget constraint, which is lower than the one required by PyTorch Autodiff,
  is achieved by re-computing some of the activations instead of storing them for gradient
  calculation.
- Depending on the budget, `Rockmate` defines automatically which activations should be recomputed.

More information on [our repository](https://github.com/topal-team/rockmate).
