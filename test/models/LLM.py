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
from transformers import LlamaModel, LlamaConfig, LlamaForSequenceClassification

def get7Bllama(batch, seq_len, nlayers=32, dtype=None, llama3=False, classification=False):
    if dtype is None:
        dtype = torch.get_default_dtype()

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
    if classification:
        model = LlamaForSequenceClassification(configuration).to(dtype)
        model.config.pad_token_id = 0
    else:
        model = LlamaModel(configuration).to(dtype)
    return model, [sample]

def get13Bllama(batch, seq_len, nlayers=40, dtype=None, llama3=False, classification=False):
    if dtype is None:
        dtype = torch.get_default_dtype()
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
    if classification:
        model = LlamaForSequenceClassification(configuration).to(dtype)
        model.config.pad_token_id = 0
    else:
        model = LlamaModel(configuration).to(dtype)

    return model, [sample]

def get3BPhi_2(batch, seq_len, dtype=None, nlayers=32, classification=False):
    from transformers import PhiModel, PhiConfig, PhiForSequenceClassification
    if dtype is None:
        dtype = torch.get_default_dtype()
    # 2.7B parameters
    configuration = PhiConfig(intermediate_size= 10240,
                            hidden_size=2560,
                            num_hidden_layers=nlayers)
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    if classification:
        model = PhiForSequenceClassification(configuration).to(dtype)
        model.config.pad_token_id = 0
    else:
        model = PhiModel(configuration).to(dtype)
    return model, [sample]
