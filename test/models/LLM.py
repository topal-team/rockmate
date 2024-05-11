"""
This file comes from :
https://amaarora.github.io/posts/2020-02-18-annotatedGPT2.html

We do not use HuggingFace GPT because torch.jit fails to trace it
"""
import torch
import math
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaConfig
from transformers.models.bloom.modeling_bloom import BloomMLP, BloomConfig
from transformers import LlamaModel, LlamaConfig
from transformers import PhiModel, PhiConfig

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
        # d_head=64,
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

    def forward(self, hidden_states):
        x = self.c_attn(hidden_states)  # new `x` shape - `[1,3,2304]`
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


class Residual(nn.Module):
    def __init__(self, layer1, layer2):
        super(Residual, self).__init__()
        self.layer1 = layer1
        self.layer2 = layer2

    def forward(self, x):
        return x + self.layer2(self.layer1(x))


def getTransformer(hidden_size=1024, intermediate_size=1024, n_head=12, dropout=0.1):
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

def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for i in range(n)])

def get_llama():
    pass


class Llama(nn.Module):
    def __init__(
        self,
        nlayers=30,
        n_ctx=2048,
        n_head=32,
        hidden_size=2560,
        # intermediate_size = 2560,
        vcb_sz=32000,
        dropout=0.1,
        embedding=False
    ):
        super(Llama, self).__init__()
        self.nlayers = nlayers
        # block = getTransformer(d_model=d_model, n_head=n_head, dropout=dropout)
        mlp = LlamaMLP(LlamaConfig(hidden_size=4096, 
                            intermediate_size=11008))
        attn = Attention(
        d_model=hidden_size,
        n_head=n_head,
        n_ctx=n_ctx,
        bias=True,
        scale=False,
        dropout=dropout,
        )
        self.hidden_size = hidden_size
        self.attens = _get_clones(attn, nlayers)
        self.mlps = _get_clones(mlp, nlayers)
        self.hidden_size = hidden_size
        self.attens = _get_clones(attn, nlayers)
        self.mlps = _get_clones(mlp, nlayers)
        self.embedding = embedding
        if self.embedding:
            self.wte = nn.Embedding(vcb_sz, hidden_size)
            self.wpe = nn.Embedding(n_ctx, hidden_size)
            self.out = nn.Linear(hidden_size, vcb_sz, bias=False)
            self.out.weight = self.wte.weight
            self.drop = nn.Dropout(dropout)
    def forward(self, inp):
        if self.embedding:
            pos_ids = torch.arange(
                0, inp.size(-1), dtype=torch.long, device=self.wpe.weight.device
            ).unsqueeze(0)
            inp = self.drop((self.wte(inp) + self.wpe(pos_ids)))

        for i in range(self.nlayers):
            inp = self.attens[i](inp)
            inp = LayerNorm(self.hidden_size)(inp)
            inp = self.mlps[i](inp)
        if self.embedding:
            inp = self.out(inp)
        return inp


class Bloom(nn.Module):
    def __init__(
        self,
        nlayers=30,
        n_ctx=2048,
        n_head=32,
        hidden_size=2560,
        # intermediate_size = 2560,
        vcb_sz=32000,
        dropout=0.1,
        embedding=False
    ):
        super(Bloom, self).__init__()
        self.nlayers = nlayers
        # block = getTransformer(d_model=d_model, n_head=n_head, dropout=dropout)
        # mlp = BloomMLP(BloomConfig(hidden_size=2560))
        mlp = nn.Sequential(nn.Linear(hidden_size, 4* hidden_size), 
                            # nn.GELU(), 
                            nn.Linear(4 * hidden_size, hidden_size))
        attn = Attention(
        d_model=hidden_size,
        n_head=n_head,
        n_ctx=n_ctx,
        bias=True,
        scale=False,
        dropout=dropout,
        )
        self.hidden_size = hidden_size
        self.attens = _get_clones(attn, nlayers)
        self.mlps = _get_clones(mlp, nlayers)
        self.embedding = embedding
        if self.embedding:
            self.wte = nn.Embedding(vcb_sz, hidden_size)
            self.wpe = nn.Embedding(n_ctx, hidden_size)
            self.out = nn.Linear(hidden_size, vcb_sz, bias=False)
            self.out.weight = self.wte.weight
            self.drop = nn.Dropout(dropout)

        self.ln = LayerNorm(self.hidden_size)
        self.init_weights()

    def forward(self, inp):
        if self.embedding:
            pos_ids = torch.arange(
                0, inp.size(-1), dtype=torch.long, device=self.wpe.weight.device
            ).unsqueeze(0)
            inp = self.drop((self.wte(inp) + self.wpe(pos_ids)))

        for i in range(self.nlayers):
            inp = self.ln(inp)
            inp_ = self.attens[i](inp)
            inp = self.ln(inp_)
            inp = self.mlps[i](inp)
        if self.embedding:
            inp = self.out(inp)
        return inp
    
    def init_weights(self):
        # self.out.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # print(module)
            module.weight.data.normal_(mean=0., std=0.2)
            if (
                isinstance(module, (nn.Linear, Conv1D))
                and module.bias is not None
            ):
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.normal_(mean=0., std=0.02)
    
# def get3Bllm(batch, seq_len, nlayers=30):
#     #https://huggingface.co/bigscience/bloom-3b#model-details
#     sample = torch.randn(batch, seq_len, 2560)
#     return Bloom(nlayers=nlayers, hidden_size=2560, n_head=32), [sample]

# def get3Bllm_embed(batch, seq_len, nlayers=30):
#     #https://huggingface.co/bigscience/bloom-3b#model-details
#     sample = torch.randint(0, 600, [batch, seq_len])
#     return Bloom(nlayers=nlayers, hidden_size=2560, n_head=32, embedding=True), [sample]

# def get7Bllm(batch, seq_len, nlayers=32):
#     #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
#     sample = torch.randn(batch, seq_len, 4096)
#     return Llama(nlayers=nlayers, hidden_size=4096, n_head=32), [sample]

# def get7Bllm_embed(batch, seq_len, nlayers=32):
#     #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
#     sample = torch.randint(0, 600, [batch, seq_len])
#     return Llama(nlayers=nlayers, hidden_size=4096, n_head=32, embedding=True), [sample]

def get7Bllama(batch, seq_len, nlayers=32, dtype=torch.float32, llama3=False):
    #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
    sample = torch.randint(0, 600, [batch, seq_len])
    # Initializing a LLaMA llama-7b style configuration
    vocab_size = 128256 if llama3 else 32000
    configuration = LlamaConfig(
                                vocab_size=vocab_size,
                                num_hidden_layers=nlayers,
                                hidden_size=4096,
                                output_hidden_states=False,
                                output_attentions=False,
                                # use_cache=False
                                )
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    # Initializing a model from the llama-7b style configuration
    model = LlamaModel(configuration)#.to(dtype)
    return model, [sample]

def get13Bllama(batch, seq_len, nlayers=40, dtype=torch.float32, llama3=False):
    #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
    sample = torch.randint(0, 600, [batch, seq_len])
    # Initializing a LLaMA llama-7b style configuration
    vocab_size = 128256 if llama3 else 32000
    configuration = LlamaConfig(
                        vocab_size=vocab_size,
                        num_hidden_layers=nlayers, 
                        hidden_size=5120,
                        intermediate_size=13824,
                        num_attention_heads=40,
                        output_hidden_states=False,
                        output_attentions=False,
                        )
    # Initializing a model from the llama-7b style configuration
    model = LlamaModel(configuration)#.to(dtype)
    return model, [sample]

def get3BPhi_2(batch, seq_len, nlayers=32):
    # 2.7B parameters
    configuration = PhiConfig(intermediate_size= 10240,
                            hidden_size=2560,
                            num_hidden_layers=nlayers)
    sample = torch.randint(0, 600, [batch, seq_len])
    model = PhiModel(configuration)
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    return model, [sample]
