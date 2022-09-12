import torch
from torch import tensor

from example_modules import GPT2
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# import random
# from rockmate.utils import *
# random.seed(0)
# torch.random.manual_seed(0)

init_code = """
___47_x = torch.addmm(model2.h[0].attn.c_attn.bias, __49_fv, model2.h[0].attn.c_attn.weight)
__47_x = ___47_x#.detach()
__51_fv = torch.Tensor.view(__47_x, [__42__0, __44__2, 2304])
__39__0 = torch.split(__51_fv, 768, 2)
__55_x1 = __39__0[2]
__80__17 = torch.Tensor.size(__55_x1, (- 1))
__76__13 = torch.Tensor.size(__55_x1, 0)
__78__15 = torch.Tensor.size(__55_x1, 1)
__82_x4 = torch.Tensor.view(__55_x1, [__76__13, __78__15, 12, torch.div(__80__17, tensor(12), rounding_mode='trunc')])
__84_v = torch.permute(__82_x4, [0, 2, 1, 3])
__53_x = __39__0[0]
__56__1 = torch.Tensor.size(__53_x, 0)
__58__3 = torch.Tensor.size(__53_x, 1)
__60__5 = torch.Tensor.size(__53_x, (- 1))
__62_x2 = torch.Tensor.view(__53_x, [__56__1, __58__3, 12, torch.div(__60__5, tensor(12), rounding_mode='trunc')])
__64_q = torch.permute(__62_x2, [0, 2, 1, 3])
__54_x0 = __39__0[1]
__66__7 = torch.Tensor.size(__54_x0, 0)
__68__9 = torch.Tensor.size(__54_x0, 1)
__70__11 = torch.Tensor.size(__54_x0, (- 1))
__72_x3 = torch.Tensor.view(__54_x0, [__66__7, __68__9, 12, torch.div(__70__11, tensor(12), rounding_mode='trunc')])
__74_k = torch.permute(__72_x3, [0, 2, 1, 3])
__87_fv = torch.transpose(__74_k, (- 2), (- 1))
"""

model2 = GPT2(nlayers=4,dropout=1e-10, vcb_sz=600).cuda()
__49_fv = torch.ones([100, 768],device=device)
__42__0, __44__2 = 10,10
exec(init_code)

mem_before = torch.cuda.memory_allocated()
# print(mem_before)
__86_scores = torch.matmul(__64_q, __87_fv)
print(torch.cuda.memory_allocated()- mem_before)
# mem_result, lb = inspection(strange_list[1], newmod.list_kg[0].sg, globals())
# print(mem_result)
