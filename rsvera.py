import torch
import random
import argparse
import numpy as np
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
from rsvera.config import VeraConfig
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoConfig
from tqdm import tqdm
from rsvera.model import VeraModel


PEFT_TYPE_TO_MODEL_MAPPING['VERA'] = VeraModel

# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)
torch.manual_seed(456)
random.seed(456)
np.random.seed(456)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="Model name or path")
parser.add_argument("--task", type=str, default="cola", help="Task name")
parser.add_argument("--peft_type", type=str, default="VERA", help="PEFT type")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
parser.add_argument("--r", type=int, default=8, help="R value for VeraConfig")
parser.add_argument("--vera_alpha", type=int, default=8, help="Vera alpha value for VeraConfig")
parser.add_argument("--use_rsvera", action="store_true", help="Whether to use RSVeRA")
parser.add_argument("--head_lr", type=float, default=4e-4, help="Learning rate (head)")
parser.add_argument("--vera_lr", type=float, default=4e-4, help="Learning rate (vera)")

args = parser.parse_args()
# Assign configuration values
batch_size = args.batch_size
model_name_or_path = args.model_name_or_path
task = args.task
peft_type = args.peft_type
device = args.device
num_epochs = args.num_epochs
max_length = args.max_length

peft_config = VeraConfig(
    task_type="SEQ_CLS", 
    inference_mode=False, 
    r=args.r, 
    vera_alpha=args.vera_alpha,
    use_rsvera=args.use_rsvera,
    projection_prng_key=0xABC,
    d_initial=0.1,
    target_modules=["key","query", "value"],
    save_projection=True
)
head_lr = args.head_lr
vera_lr = args.vera_lr

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
    if task == "sst2":
        return tokenizer(examples["sentence"], truncation=True, max_length=max_length)
    elif task == "mrpc":
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    elif task == "cola":
        return tokenizer(examples["sentence"], truncation=True, max_length=max_length)
    elif task == "qnli":
        return tokenizer(examples["question"], examples["sentence"], truncation=True, max_length=max_length)
    elif task == "rte":
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    elif task == "stsb":
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    else:
        raise ValueError(f"Task {task} not supported.")

def remove_columns(task):
    if task == "sst2":
        return ["idx", "sentence"]
    elif task == "mrpc":
        return ["idx", "sentence1", "sentence2"]
    elif task == "cola":
        return ["idx", "sentence"]
    elif task == "qnli":
        return ["idx", "question", "sentence"]
    elif task == "rte":
        return ["idx", "sentence1", "sentence2"]
    elif task == "stsb":
        return ["idx", "sentence1", "sentence2"]
    else:
        raise ValueError(f"Task {task} not supported.")

tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=remove_columns(task),
)

# We also rename the 'label' column to 'labels' which is the expected name for labels by the models of the
# transformers library
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")


def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

def relabel(example):
    example['labels']=1 
    return example

tokenized_datasets["test"] = tokenized_datasets["test"].map(relabel)


# Instantiate dataloaders.
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
test_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

if task == "stsb":
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None, num_labels = 1)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model

optimizer = AdamW(
    [
        {"params": [p for n, p in model.named_parameters() if "vera_lambda_" in n], "lr": vera_lr},
        #{"params": [p for n, p in model.named_parameters() if "classifier" in n], "lr": head_lr},
    ]
)

# Instantiate scheduler
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_dataloader) * num_epochs),
    num_training_steps=(len(train_dataloader) * num_epochs),
)

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
    
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        batch.to(device)
        with torch.no_grad():
            outputs = model(**batch)
        if task == "stsb":
            predictions = outputs.logits
        else:
            predictions = outputs.logits.argmax(dim=-1)
        predictions, references = predictions, batch["labels"]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    
    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)
    
    
save_model(model, f"{model_name_or_path}_{task}.safetensors")
load_model(model, f"{model_name_or_path}_{task}.safetensors")
#print(model.state_dict())

# peft_model_id = "afmck/roberta-large-peft-vera"
# config = PeftConfig.from_pretrained(peft_model_id)
#inference_model = AutoModelForSequenceClassification.from_pretrained(peft_config.base_model_name_or_path)
#tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Load the Vera model
#inference_model = PeftModel.from_pretrained(inference_model, model)
#print("testt:", test_dataloader)
model.to(device)
model.eval()
for step, batch in enumerate(tqdm(eval_dataloader)):
    batch.to(device)
    with torch.no_grad():
        outputs = model(**batch)
    if task == "stsb":
        predictions = outputs.logits
    else:
        predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )
#print("testtt:", test_dataloader)

eval_metric = metric.compute()
print("Final evaluate result: ", eval_metric)