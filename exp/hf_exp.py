import pickle
# import deepspeed
import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
# from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from exp import LlamaConfig

from transformers import AutoTokenizer, LlamaTokenizerFast
from transformers import Trainer
from LLM import get7Bllama, get3BPhi_2, get13Bllama, get7Bllama_lora
from datetime import datetime
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        # self.encodings = {"input_ids":[torch.randint(0, 600, [batch, seq_len])
        #                                for _ in range(30)]}
        self.labels = [1 for _ in encodings]
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) 
                for key, val in self.encodings.items()
                if "input_ids" in key}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings)
    

# from transformers import LlamaForSequenceClassification
# def get7Bllama(batch, seq_len, nlayers=32, dtype=torch.float32):
#     if dtype is None:
#         dtype = torch.get_default_dtype()
#     #https://huggingface.co/docs/transformers/main/model_doc/llama2#transformers.LlamaConfig
#     sample = torch.randint(0, 600, [batch, seq_len])
#     # Initializing a LLaMA llama-7b style configuration
#     configuration = LlamaConfig(num_hidden_layers=nlayers,
#                                 hidden_size=4096,
#                                 output_hidden_states=False,
#                                 output_attentions=False,
#                                 pad_token_id=0,
#                                 use_cache=False
#                                 )
#     # Initializing a model from the llama-7b style configuration
#     model = LlamaForSequenceClassification(configuration).to(dtype)
#     # model = LlamaModel(configuration)#.to(dtype)
#     return model, [sample]

def train_po(nlayers=4, batch =4, seq_len=512, get_model=get7Bllama):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer.batch_encode_plus(["a "*510]*20)#Fake data
    train_set = MyDataset(encoding)

    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.memory_allocated()
    model,_ = get_model(4, seq_len=seq_len, nlayers=nlayers, classification=True)
    training_arguments = TrainingArguments(
    output_dir="/",
    num_train_epochs=1,
    per_device_train_batch_size=batch,
    gradient_accumulation_steps=1,
    optim='paged_adamw_32bit',
    # save_steps=1,
    # logging_steps=1,
    learning_rate=1e-4,
    weight_decay=0,
    )
    trainer = Trainer(model, 
                    args=training_arguments, 
                    train_dataset=train_set,
                    #   compute_metrics=compute_metrics,
                    )
    train_result = trainer.train()
    print(nlayers, train_result.metrics["train_samples_per_second"])
    results = {}
    results["nlayers"] = nlayers
    results["input_shape"] = (batch, seq_len)
    results["peak_mem"] = torch.cuda.max_memory_allocated() - mem
    results["time"] = train_result.metrics['train_runtime'] *1000
    results["time_per_sample"] = train_result.metrics["train_samples_per_second"] *1000
    results["metrics"] = train_result.metrics
    return results
    


def train_ds(nlayers=4, batch =4, seq_len=512, 
             ds_config="ds_config_zero3.json", 
             get_model=get7Bllama):
    import deepspeed
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    encoding = tokenizer.batch_encode_plus(["a "*510]*20)#Fake data
    train_set = MyDataset(encoding)

    # tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")
    # encoding = tokenizer.batch_encode_plus(["a "*510]*20)#Fake data
    # # encdoin
    # train_set = MyDataset(encoding)
    
    torch.cuda.reset_peak_memory_stats()
    mem = torch.cuda.memory_allocated()
    # with deepspeed.zero.Init():
    model,_ = get_model(batch=batch, seq_len=seq_len, nlayers=nlayers, classification=True)
    training_arguments = TrainingArguments(
    output_dir="/",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    # optim='adamw_torch',
    deepspeed=ds_config,
    # save_steps=1,
    # logging_steps=1,
    learning_rate=1e-4,
    weight_decay=0,
    )
    # model.gradient_checkpointing_enable()
    # def make_inputs_require_grad(module, input, output):
    #      output.requires_grad_(True)

    # model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    trainer = Trainer(model, 
                        args=training_arguments, 
                        train_dataset=train_set,
                        #   compute_metrics=compute_metrics,
                        )
    train_result = trainer.train()

    print(nlayers, train_result.metrics["train_samples_per_second"])
    results = {}
    results["nlayers"] = nlayers
    results["input_size"] = (4, seq_len)
    results["peak_mem"] = torch.cuda.max_memory_allocated() - mem
    results["time"] = train_result.metrics['train_runtime'] *1000
    results["time_per_sample"] = train_result.metrics["train_samples_per_second"]*1000
    results["metrics"] = train_result.metrics
    results["gpu_type"] = torch.cuda.get_device_name()
    results["date"] = datetime.now().strftime("%x_%H:%M")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", nargs="?",  type=str, default="0")
    parser.add_argument("--method", nargs="?",  type=str, default="zero-3")
    parser.add_argument("--nlayers", nargs="?",  default=12, type=int)
    parser.add_argument("--batch_size", nargs="?", default=4, type=int)
    parser.add_argument("--dtype", nargs="?", type=str, default="float32")
    parser.add_argument("--model", nargs="?", type=str, default="llama7b")
    
    methods = {
        "zero-3":train_ds,
        "zero-2":train_ds,
        "paged_optim":train_po,
               }
    dtypes = {"float32":torch.float32,
              "bfloat16": torch.bfloat16,
              "float16": torch.float16,
              }
    
    models = {
        "llama7b": get7Bllama,
        "llama13b": get13Bllama,
        "phi2-3b": get3BPhi_2,
        "llama7b_lora": get7Bllama_lora,
    }

    args = parser.parse_args()
    if "zero" in args.method:
        import deepspeed
    kwargs = {}
    if "zero-3" in args.method:
        kwargs["ds_config"]="ds_config_zero3.json"
    if "zero-2" in args.method:
        kwargs["ds_config"]="ds_config_zero2.json"
    torch.set_default_dtype(dtypes[args.dtype])
    exp_id = args.exp_id
    nlayers = args.nlayers
    
    exp_id = f"exp_results/{args.method}-{args.model}-{args.dtype}-{args.exp_id}.pkl"
    if os.path.exists(exp_id):
        with open(exp_id, "rb") as f:
            results = pickle.load(f)
    else:
        results = {}
    try:
        results[nlayers] = methods[args.method](nlayers, 
                                                     batch=4, 
                                                     get_model=models[args.model],
                                                     **kwargs)
    except Exception as e:
        results[nlayers] = str(e)
    with open(exp_id, "wb") as f:
        pickle.dump(results, f)
    print(results)
