import torch
import wandb
import random
from torch.optim import AdamW
from safetensors.torch import load_model, save_model
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
from peft import (
    get_peft_model,
    #VeraConfig,
    PeftType,
)
from verac.config import VeraConfig
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoConfig
from tqdm import tqdm
from verac.model import VeraModel
from verac.layer import VeraLayer


PEFT_TYPE_TO_MODEL_MAPPING['VERA'] = VeraModel

task_dict = {
    0:"sst2",
    1:"mrpc",
    2:"cola",
    3:"qnli",
    4:"rte",
    5:"stsb",
}

batch_size = 64
model_name_or_path = "roberta-base"
task = task_dict.get(1)
peft_type = "VERA"
device = "cuda"
num_epochs = 30
max_length = 512

peft_config = VeraConfig(
    task_type="SEQ_CLS", 
    inference_mode=False, 
    r=1024, 
    projection_prng_key=0xABC,
    d_initial=0.1,
    c_initial=0.1,
    target_modules=["key","query", "value"],
    save_projection=True
)

head_lr = 4e-3
vera_lr = 1e-2

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="verac-project",
    
#     # track hyperparameters and run metadata
#     config={
#     "batch_size": 64,
#     "model_name_or_path": "Roberta_base",
#     "task": task_dict.get(1),
#     "num_epochs": 25,
#     }
# )

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    return outputs

#def tokenize_function(task, examples):
#    if task == "sst2":
#        return tokenizer(examples["sentence"], truncation=True, max_length=max_length)
#    elif task == "mrpc":
#        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
#    elif task == "cola":
#        return tokenizer(examples["sentence"], truncation=True, max_length=max_length)
#    elif task == "qnli":
#        return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=max_length)
#    elif task == "rte":
#        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
#    elif task == "stsb":
#        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
#    else:
#        raise ValueError(f"Task {task} not supported.")


#def get_remove_columns(task):
#    if task == "sst2":
#        return ["idx", "sentence"]
#    elif task == "mrpc":
#        return ["idx", "sentence1", "sentence2"]
#    elif task == "cola":
#        return ["idx", "sentence"]
#    elif task == "qnli":
#        return ["idx", "question", "sentence"]
#    elif task == "rte":
#        return ["idx", "sentence1", "sentence2"]
#    elif task == "stsb":
#        return ["idx", "sentence1", "sentence2"]
#    else:
#        raise ValueError(f"Task {task} not supported.")

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns= ["idx", "sentence1", "sentence2"],
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

#ensure all examples within a batch have the same length by padding them appropriately, 
#and convert them into PyTorch tensors
def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def relabel(example):
    example['labels']=1 
    return example

tokenized_datasets["test"] = tokenized_datasets["test"].map(relabel)


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
test_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model
#print(model)
optimizer = AdamW(
    [
        {"params": [p for n, p in model.named_parameters() if "vera_lambda_" in n], "lr": vera_lr},
        {"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": head_lr},
    ]
)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)
print("train:", train_dataloader)
model.to(device)
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch.to(device)
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
    #print("eval: ", eval_dataloader)
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
    #wandb.log({f"epoch {epoch}:": eval_metric})
    
    
save_model(model, f"{model_name_or_path}_{task}.safetensors")
load_model(model, f"{model_name_or_path}_{task}.safetensors")

model.to(device)
model.eval()



for step, batch in enumerate(tqdm(eval_dataloader)):
    batch.to(device)
    with torch.no_grad():
        outputs = model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )

eval_metric = metric.compute()
print(eval_metric)

# wandb.log(eval_metric)
# wandb.finish()

