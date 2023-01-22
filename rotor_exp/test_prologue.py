import torch
import rotor
import torch.nn as nn


def make_seq(): 
    seq = nn.Sequential(*(nn.Conv1d(1, 1, 3) for _ in range(10)))

    ## Freeze the first 3 layers, and layer 7
    for m in seq[:3]:
         for p in m.parameters():
             p.requires_grad = False

    m = seq[7]
    for p in m.parameters():
        p.requires_grad = False

    return seq


x = torch.randn([1, 1, 100])

print("No ckpt test")
seq = make_seq()
y = seq(x).sum()
print(y)
y.backward()
print(x.requires_grad, x.grad)
for i in range(10):
    print(i, any(p.grad is not None for p in seq[i].parameters()))

print("Seq pass")
seq = make_seq()
x = torch.randn([1, 1, 100])
import torch.utils.checkpoint as ckpt
y = ckpt.checkpoint_sequential(seq, 3, x).sum()
print(y)
y.backward()
print(x.requires_grad, x.grad)
for i in range(10):
    print(i, any(p.grad is not None for p in seq[i].parameters()))


from rotor.algorithms.sequence import *

print("Rotor pass")
seq = make_seq()
chk = rotor.Checkpointable(seq, mem_limit=1024*1024*1024, verbosity=5)
chk.measure(x)
core_length = len(chk.functions)
sequence = Sequence(Function("HardCoded"))
for i in range(core_length-1):
    sequence.insert(ForwardEnable(i))
sequence.insert(ForwardCheck(core_length-1))
sequence.insert(Loss())
sequence.insert(ForwardEnable(core_length-1))
sequence.insert(Backward(core_length-1))
for i in reversed(range(core_length-1)):
    sequence.insert(Backward(i))
chk.sequence = sequence
print(chk.sequence)
print(x.requires_grad)
y = chk(x).sum()
print(y)
y.backward()
print(x.requires_grad, x.grad)
for i in range(10):
    print(i, any(p.grad is not None for p in seq[i].parameters()))




# This shows that the return of a Function whose Tensor arguments do not require grad
# can not be used for backward()
# print("Test Linear")
# x2 = torch.randn([1, 100])
# w = torch.randn([1, 100], requires_grad=True)
# y = LinearFunction.apply(x2, (w, )).sum()
# print(y)
# y.backward()

# class LinearFunction(torch.autograd.Function):

#     # Note that both forward and backward are @staticmethods
#     @staticmethod
#     # bias is an optional argument
#     def forward(ctx, input, hidden_weight, bias=None):
#         weight = hidden_weight[0]
#         ctx.save_for_backward(input, weight, bias)
#         output = input.mm(weight.t())
#         if bias is not None:
#             output += bias.unsqueeze(0).expand_as(output)
#         return output

#     # This function has only a single output, so it gets only one gradient
#     @staticmethod
#     def backward(ctx, grad_output):
#         # This is a pattern that is very convenient - at the top of backward
#         # unpack saved_tensors and initialize all gradients w.r.t. inputs to
#         # None. Thanks to the fact that additional trailing Nones are
#         # ignored, the return statement is simple even when the function has
#         # optional inputs.
#         input, weight, bias = ctx.saved_tensors
#         grad_input = grad_weight = grad_bias = None

#         # These needs_input_grad checks are optional and there only to
#         # improve efficiency. If you want to make your code simpler, you can
#         # skip them. Returning gradients for inputs that don't require it is
#         # not an error.
#         if ctx.needs_input_grad[0]:
#             grad_input = grad_output.mm(weight)
#         if ctx.needs_input_grad[1]:
#             grad_weight = grad_output.t().mm(input)
#         if bias is not None and ctx.needs_input_grad[2]:
#             grad_bias = grad_output.sum(0)

#         return grad_input, grad_weight, grad_bias
