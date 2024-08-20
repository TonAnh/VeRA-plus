import torch
import random
import argparse
import numpy as np
import os
import time
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
from verasmoe.config import VeraConfig # Custom Config of Vera
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoConfig
from tqdm import tqdm
from verasmoe.model import VeraModel # Custom Vera Model


PEFT_TYPE_TO_MODEL_MAPPING['VERA'] = VeraModel # Change the VERA model in mapping to Custom Vera Model

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--output_dir", type=str, default="output", help="The output directory to save the fintune model")
parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="Model name or path")
parser.add_argument("--task", type=str, default="cola", help="Task name")
parser.add_argument("--peft_type", type=str, default="VERA", help="PEFT type")
parser.add_argument("--device", type=str, default="cuda", help="Device")
parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
parser.add_argument("--r", type=int, default=8, help="R value for VeraConfig")
parser.add_argument("--vera_alpha", type=int, default=8, help="Vera alpha value for VeraConfig")
parser.add_argument("--use_rsvera", type=bool, default=True, help="Whether to use RSVeRA")
parser.add_argument("--head_lr", type=float, default=4e-4, help="Learning rate (head)")
parser.add_argument("--vera_lr", type=float, default=4e-4, help="Learning rate (vera)")
parser.add_argument("--d_init", type=float, default=0.1, help="Initial init value for `vera_lambda_d` vector in VeRA")
parser.add_argument("--seed", type=int, default=42, help="Seed")
parser.add_argument("--num_experts", type=int, default=4, help="Number of Experts when using SMoE")
parser.add_argument("--top_k", type=int, default=1, help="Top-k experts to use")

args = parser.parse_args()

# Set seed 
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

# == Assign configuration values == 
batch_size = args.batch_size
output_dir = args.output_dir
model_name_or_path = args.model_name_or_path
task = args.task
peft_type = args.peft_type
device = args.device
num_epochs = args.num_epochs
max_length = args.max_length
head_lr = args.head_lr
vera_lr = args.vera_lr

# == Set the padding side base on model == 
if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

preprocess_time_start = time.perf_counter()
# == TOKENIZER & LOAD THE DATASET & PREPROCESS DATA == 
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

# def relabel(example):
#     example['labels']=1 
#     return example

# tokenized_datasets["test"] = tokenized_datasets["test"].map(relabel)

# == Instantiate dataloaders ==
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size
)
test_dataloader = DataLoader(tokenized_datasets["test"], shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
preprocess_time_end = time.perf_counter()

if task == "stsb":
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None, num_labels = 1)
else:
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)


# == SETUP THE MODEL == 
peft_config = VeraConfig(
    task_type="SEQ_CLS", 
    inference_mode=False, 
    r=args.r, 
    vera_alpha=args.vera_alpha,
    use_rsvera=args.use_rsvera,
    projection_prng_key=0xABC,
    d_initial=args.d_init,
    target_modules=["key","query", "value"],
    save_projection=True,
    num_experts = args.num_experts,
    top_k = args.top_k,
)

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

training_time_start = time.perf_counter()
# == TRAINING LOOP == 
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

training_time_end = time.perf_counter()

# Save the fintune model
save_path = f"{output_dir}/{model_name_or_path}_{task}.safetensors"

# Check if the directory exist, if not -> create it
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

save_model(model, save_path)

# Load model for evaluate
load_model(model, save_path)
evaluate_time_start = time.perf_counter()
# == EVALUATE LOOP == 
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

valid_eval_metric = metric.compute()
print("Final `validation` evaluate result: ", eval_metric)
evaluate_time_end = time.perf_counter()

# ==== TESTING LOOP =====
for step, batch in enumerate(tqdm(test_dataloader)):
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

test_eval_metric = metric.compute()
print("Final `testing` evaluate result: ", test_eval_metric)

# Calculate run time
preprocess_time = preprocess_time_end - preprocess_time_start
training_time = training_time_end - training_time_start
evaluate_time = evaluate_time_end - evaluate_time_start

print("==== RUNTIME ====")
print(f"Total run time: {(preprocess_time + training_time + evaluate_time):.2f}s")
print(f"Preprocess time: {(preprocess_time):.2f}s")
print(f"Training time: {(training_time):.2f}s")
print(f"Evaluate time: {(evaluate_time):.2f}s")