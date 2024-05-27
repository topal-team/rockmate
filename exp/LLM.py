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
# from transformers.models.llama.modeling_llama import LlamaMLP, LlamaConfig
# from transformers.models.bloom.modeling_bloom import BloomMLP, BloomConfig
from transformers import LlamaModel, LlamaConfig, LlamaForSequenceClassification
from transformers import BloomConfig, BloomForSequenceClassification
from peft import LoraModel, LoraConfig

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # self.encodings = {"input_ids":[torch.randint(0, 600, [batch, seq_len])
        #                                for _ in range(30)]}
        self.labels = [1 for _ in encodings]
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()
                # if "input_ids" in key
                }
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)

def get7Bllama(batch, seq_len, nlayers=32, dtype=None, llama3=False, classification=False):
    if dtype is None:
        dtype = torch.get_default_dtype()

    #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
    sample = torch.randint(0, 600, [batch, seq_len])
    # Initializing a LLaMA llama-7b style configuration
    vocab_size = 128256 if llama3 else 32000
    

    configuration = LlamaConfig(num_hidden_layers=nlayers,
                                hidden_size=4096,
                                output_hidden_states=False,
                                output_attentions=False,
                                pad_token_id=0,
                                use_cache=False
                                )
    # Initializing a model from the llama-7b style configuration
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    model = LlamaForSequenceClassification(configuration).to(dtype)
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
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    # Initializing a model from the llama-7b style configuration
    # if classification:
    configuration = LlamaConfig(
                    vocab_size=vocab_size,
                    num_hidden_layers=nlayers, 
                    hidden_size=5120,
                    intermediate_size=13824,
                    num_attention_heads=40,
                    output_hidden_states=False,
                    output_attentions=False,
                    pad_token_id=0,
                    use_cache=False
                    )
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    model = LlamaForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    # else:
    #     model = LlamaModel(configuration).to(dtype)

    return model, [sample]

def get3BPhi_2(batch, seq_len, dtype=None, nlayers=32, classification=False):
    from transformers import PhiModel, PhiConfig, PhiForSequenceClassification
    if dtype is None:
        dtype = torch.get_default_dtype()
    # 2.7B parameters
    configuration = PhiConfig(intermediate_size= 10240,
                            hidden_size=2560,
                            num_hidden_layers=nlayers,
                            output_hidden_states=False,
                            output_attentions=False,
                            pad_token_id=0,
                            use_cache=False)
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    # if classification:
    model = PhiForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    # else:
    #     model = PhiModel(configuration).to(dtype)
    return model, [sample]



def get3BPhi_15(batch, seq_len, dtype=None, nlayers=24, classification=False):
    from transformers import PhiModel, PhiConfig, PhiForSequenceClassification
    if dtype is None:
        dtype = torch.get_default_dtype()
    # 2.7B parameters
    configuration = PhiConfig(intermediate_size= 8192,
                            hidden_size=2048,
                            num_hidden_layers=nlayers,
                            output_hidden_states=False,
                            output_attentions=False,
                            pad_token_id=0,
                            use_cache=False)
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    # if classification:
    model = PhiForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    # else:
    #     model = PhiModel(configuration).to(dtype)
    return model, [sample]


class LoraLinear(nn.Module):
    def __init__(self, linear, num_adapters=10, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        u = nn.Parameter(torch.randn(num_adapters, self.linear.weight.shape[0]), requires_grad=True)
        v = nn.Parameter(torch.randn(self.linear.weight.shape[1], num_adapters), requires_grad=True)
        self.register_parameter("u", u)
        self.register_parameter("v", v)

    def forward(self, x):
        res1 = torch.matmul(x, self.v)
        res2 = torch.matmul(res1, self.u)
        y = self.linear(x)
        out = y+res2
        return out

def manual_lora(model:nn.Module, target_modules, num_adapters=10, freeze_all=True):
    if freeze_all:
        for p in model.parameters():
            p.requires_grad = False
    for module_name in target_modules:
        module = model.get_submodule(module_name)
        if isinstance(module, nn.Linear):
            new_module = LoraLinear(module, num_adapters=num_adapters)
        else:
            raise TypeError(f"manual lora does not work with {type(module)}")
        # setattr(model, module_name, new_module)

        atoms = module_name.split(".")
        mod: torch.nn.Module = model

        for item in atoms:
            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                        "attribute `" + item + "`")
            if getattr(mod, item) == module:
                # setattr(mod, item, new_module)
                mod.add_module(item, new_module)
                break

            mod = getattr(mod, item)
            if not isinstance(mod, torch.nn.Module):
                raise AttributeError("`" + item + "` is not "
                                        "an nn.Module")

def get7Bllama_lora(batch, seq_len, num_adapters=64, nlayers=32, dtype=None, llama3=False, classification=False):
    model, sample = get7Bllama(batch, 512, nlayers=nlayers, dtype=dtype, llama3=llama3, classification=classification)
    config = LoraConfig(
        r=num_adapters,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj"],
        lora_dropout=0.01,
    )
    model = LoraModel(model, config, "default")

    # target_modules = [f"layers.{i}.self_attn.q_proj" for i in range(nlayers)]
    # target_modules += [f"layers.{i}.self_attn.k_proj" for i in range(nlayers)]
    # target_modules += [f"layers.{i}.self_attn.v_proj" for i in range(nlayers)]
    # # if classification:
    # target_modules = ["model."+s for s in target_modules]
    # manual_lora(model, 
    #             target_modules=target_modules,
    #             num_adapters=num_adapters,
    #             freeze_all=True)
    if classification:
        model.enable_input_require_grads()

    return model, sample


def get11Bfalcon(batch, seq_len, dtype=None, nlayers=24, classification=False):
    from transformers import FalconForSequenceClassification, FalconConfig
    if dtype is None:
        dtype = torch.get_default_dtype()
        config = {
                                    "ffn_hidden_size": 16384,
                                    "hidden_dropout": 0.0,
                                    "hidden_size": 4096,
                                    "num_attention_heads": 32,
                                    "num_hidden_layers":nlayers,
                                    "torch_dtype": "bfloat16",
                                    "use_cache": False,
                                    "output_hidden_states":False,
                                    "output_attentions":False,
                                    "vocab_size": 65024
        }
        configuration = FalconConfig( **config
                            )
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    # if classification:
    model = FalconForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    return model, [sample]


def get3Bbloom(batch, seq_len, dtype=None, nlayers=30, classification=False):
    from transformers import BloomForSequenceClassification, BloomConfig
    if dtype is None:
        dtype = torch.get_default_dtype()
    configuration = BloomConfig(hidden_size=2560,
                            num_hidden_layers=nlayers,
                            output_hidden_states=False,
                            output_attentions=False,
                            pad_token_id=0,
                            use_cache=False
                            )
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    # if classification:
    model = BloomForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    return model, [sample]

def get8Bllama(batch, seq_len, nlayers=32, dtype=None, llama3=True):
    if dtype is None:
        dtype = torch.get_default_dtype()

    #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
    sample = torch.randint(0, 600, [batch, seq_len])
    # Initializing a LLaMA llama-7b style configuration
    vocab_size = 128256 if llama3 else 32000
    

    configuration = LlamaConfig(num_hidden_layers=nlayers,
                                hidden_size=4096,
                                output_hidden_states=False,
                                output_attentions=False,
                                pad_token_id=0,
                                use_cache=False,
                                vocab_size=vocab_size
                                )
    # Initializing a model from the llama-7b style configuration
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    model = LlamaForSequenceClassification(configuration).to(dtype)
    return model, [sample]

def get7Bmistral(batch, seq_len, dtype=None, nlayers=32, classification=False):
    from transformers import MistralConfig, MistralForSequenceClassification
    if dtype is None:
        dtype = torch.get_default_dtype()
    configuration = MistralConfig(hidden_size=4096,
                            num_hidden_layers=nlayers,
                            num_attention_heads=32,
                            intermediate_size=14336,
                            output_hidden_states=False,
                            output_attentions=False,
                            pad_token_id=0,
                            use_cache=False,
                            vocab_size= 32000
                            )
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    # if classification:
    model = MistralForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    return model, [sample]

def get4Bphi3(batch, seq_len, dtype=None, nlayers=32, classification=False):
    from transformers import PhiConfig, PhiForSequenceClassification
    if dtype is None:
        dtype = torch.get_default_dtype()
    configuration = PhiConfig(hidden_size=3072,
                            num_hidden_layers=nlayers,
                            num_attention_heads=32,
                            intermediate_size=8192,
                            output_hidden_states=False,
                            output_attentions=False,
                            pad_token_id=0,
                            use_cache=False
                            )
    configuration._attn_implementation="eager"
    configuration._attn_implementation_internal="eager"
    sample = torch.randint(0, 600, [batch, seq_len])
    # if classification:
    model = PhiForSequenceClassification(configuration).to(dtype)
    model.config.pad_token_id = 0
    return model, [sample]
