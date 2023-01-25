import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class FeedForward(nn.Module):
    def __init__(self, dropout, d_model=768, nx=768 * 4):
        super().__init__()
        self.c_fc = Conv1D(d_model, nx)
        self.c_proj = Conv1D(nx, d_model)
        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Identity()

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class Attention(nn.Module):
    def __init__(
        self,
        d_model=768,
        n_head=12,
        n_ctx=1024,
        d_head=64,
        bias=True,
        scale=False,
        dropout=0.1,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.c_attn = Conv1D(d_model, d_model * 3)
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx),
        )
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Identity()
        self.c_proj = Conv1D(d_model, d_model)

    def split_heads(self, x):
        "return shape [`batch`, `head`, `sequence`, `features`]"
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def _attn(self, q, k, v, attn_mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.scale:
            scores = scores / math.sqrt(v.size(-1))
        nd, ns = scores.size(-2), scores.size(-1)
        if attn_mask is not None:
            scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(new_shape)

    def forward(self, x):
        x = self.c_attn(x)  # new `x` shape - `[1,3,2304]`
        q, k, v = x.split(self.d_model, dim=2)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        # out      = self._attn(q, k, v)
        scores = torch.matmul(q, k.transpose(-2, -1))
        # if self.scale: scores = scores/math.sqrt(v.size(-1))
        nd, ns = scores.size(-2), scores.size(-1)
        # if attn_mask is not None: scores = scores + attn_mask
        scores = self.softmax(scores)
        scores = self.dropout(scores)
        out = torch.matmul(scores, v)

        out = self.merge_heads(out)
        out = self.c_proj(out)
        return out


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model=768, n_head=12, dropout=0.1):
#         super(TransformerBlock, self).__init__()
#         self.attn = Attention(
#             d_model=d_model,
#             n_head=n_head,
#             d_head=64,
#             n_ctx=1024,
#             bias=True,
#             scale=False,
#             dropout=dropout,
#         )
#         self.feedforward = FeedForward(
#             dropout=dropout, d_model=d_model, nx=d_model * 4
#         )
#         self.ln_1 = LayerNorm(d_model)
#         self.ln_2 = LayerNorm(d_model)

#     def forward(self, x):
#         x1 = self.ln_1(x)
#         x2 = self.ln_2(x)
#         x = x + self.attn(x1)
#         x = x + self.feedforward(x2)
#         return x


class Residual(nn.Module):
    def __init__(self, layer1, layer2):
        super(Residual, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        return x + self.layer2(self.layer1(x))


def getTransformer(d_model=768, n_head=12, dropout=0.1):
    attn = Attention(
        d_model=d_model,
        n_head=n_head,
        d_head=64,
        n_ctx=1024,
        bias=True,
        scale=False,
        dropout=dropout,
    )
    feedforward = FeedForward(dropout=dropout, d_model=d_model, nx=d_model * 4)
    ln1 = LayerNorm(d_model)
    ln2 = LayerNorm(d_model)
    return nn.Sequential(*[Residual(ln1, attn), Residual(ln2, feedforward)])


class TransformerBlock(nn.Module):
    def __init__(self, d_model=768, n_head=12, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = Attention(
            d_model=d_model,
            n_head=n_head,
            d_head=64,
            n_ctx=1024,
            bias=True,
            scale=False,
            dropout=dropout,
        )
        self.feedforward = FeedForward(
            dropout=dropout, d_model=d_model, nx=d_model * 4
        )
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)

    def forward(self, x):
        x1 = self.ln_1(x)
        x2 = self.ln_2(x)
        x = x + self.attn(x1)
        x = x + self.feedforward(x2)
        return x


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for i in range(n)])


class GPT2(nn.Module):
    def __init__(
        self,
        nlayers=12,
        n_ctx=2048,
        n_head=12,
        d_model=768,
        vcb_sz=50257,
        dropout=0.1,
    ):
        super(GPT2, self).__init__()
        self.nlayers = nlayers
        # block = TransformerBlock(
        #     d_model=d_model, n_head=n_head, dropout=dropout
        # )
        block = getTransformer(d_model=d_model, n_head=n_head, dropout=dropout)
        self.h = _get_clones(block, nlayers)
        self.wte = nn.Embedding(vcb_sz, d_model)
        self.wpe = nn.Embedding(n_ctx, d_model)
        self.drop = nn.Dropout(dropout)
        # self.drop = nn.Identity()

        self.ln_f = LayerNorm(d_model)
        self.out = nn.Linear(d_model, vcb_sz, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()
        self.n_head = n_head
        self.d_model = d_model

    def init_weights(self):
        self.out.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if (
                isinstance(module, (nn.Linear, Conv1D))
                and module.bias is not None
            ):
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, src, labels=None, pos_ids=None, return_inp=False, dropout=0.1
    ):
        if pos_ids is None:
            pos_ids = torch.arange(
                0, src.size(-1), dtype=torch.long, device=self.wpe.weight.device
            ).unsqueeze(0)
        inp = self.drop((self.wte(src) + self.wpe(pos_ids)))
        if return_inp:
            return inp
        for i in range(self.nlayers):
            inp = self.h[i](inp)
        inp = self.ln_f(inp)
        logits = self.out(inp)
        outputs = (logits,) + (inp,)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            outputs = (loss,) + outputs
            return outputs
        return logits


class GPT2_input(nn.Module):
    def __init__(self, n_ctx=1024, d_model=768, vcb_sz=50257, dropout=0.1):
        super(GPT2_input, self).__init__()
        block = TransformerBlock(d_model=d_model, n_head=12, dropout=dropout)
        # self.h = _get_clones(block, nlayers)
        self.wte = nn.Embedding(vcb_sz, d_model)
        self.wpe = nn.Embedding(n_ctx, d_model)
        self.drop = nn.Dropout(dropout)
        # self.drop = nn.Identity()
        self.ln_f = LayerNorm(d_model)
        self.out = nn.Linear(d_model, vcb_sz, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        self.out.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if (
                isinstance(module, (nn.Linear, Conv1D))
                and module.bias is not None
            ):
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, src, labels=None, pos_ids=None, return_inp=False, dropout=0.1
    ):
        if pos_ids is None:
            pos_ids = torch.arange(
                0, src.size(-1), dtype=torch.long, device=self.wpe.weight.device
            ).unsqueeze(0)
        inp = self.drop((self.wte(src) + self.wpe(pos_ids)))
        return inp


class GPT2_output(nn.Module):
    def __init__(self, d_model=768, vcb_sz=50257, dropout=0.1):
        super(GPT2_output, self).__init__()
        # self.nlayers = nlayers
        # block = TransformerBlock(d_model=d_model, n_head=12, dropout=dropout)
        # self.h = _get_clones(block, nlayers)
        self.wte = nn.Embedding(vcb_sz, d_model)
        # self.wpe = nn.Embedding(n_ctx, d_model)
        self.drop = nn.Dropout(dropout)
        # self.drop    = nn.Identity()
        self.ln_f = LayerNorm(d_model)
        self.out = nn.Linear(d_model, vcb_sz, bias=False)
        self.loss_fn = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        self.out.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if (
                isinstance(module, (nn.Linear, Conv1D))
                and module.bias is not None
            ):
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self, inp, labels=None, pos_ids=None, return_inp=False, dropout=0.1
    ):
        inp = self.ln_f(inp)
        logits = self.out(inp)
        outputs = (logits,) + (inp,)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = self.loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            outputs = (loss,) + outputs
            return outputs
        return logits


def get_GPT(model="GPT2-small"):
    if model == "GPT2-small":  # 117M parameters
        return GPT2(nlayers=12, d_model=768, n_head=12)
    if model == "GPT2-medium":  # 345M parameters
        return GPT2(nlayers=24, d_model=1024, n_head=16)
    if model == "GPT2-large":  # 774M parameters
        return GPT2(nlayers=36, d_model=1280, n_head=20)
    if model == "GPT2-xl":  # 1.6B parameters
        return GPT2(nlayers=48, d_model=1600, n_head=25)

    if model == "GPT3-small":  # 125M parameters
        return GPT2(nlayers=12, d_model=768, n_head=12)
    if model == "GPT3-medium":  # 350M parameters
        return GPT2(nlayers=24, d_model=1024, n_head=16)
    if model == "GPT3-large":  # 760M parameters
        return GPT2(nlayers=24, d_model=1536, n_head=16)
    if model == "GPT3-xl":  # 1.3B parameters
        return GPT2(nlayers=24, d_model=2048, n_head=24)

