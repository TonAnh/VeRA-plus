import torch
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
from vera.config import VeraConfig
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed, AutoConfig
from tqdm import tqdm
from vera.model import VeraModel


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
    target_modules=["key","query", "value"],
    save_projection=True
)
head_lr = 4e-3
vera_lr = 1e-2

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side=padding_side)
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

datasets = load_dataset("glue", task)
metric = evaluate.load("glue", task)

# def get_dataset(ds_name, task, split="train"):
#     dataset = load_dataset(ds_name, task, split=split)
#     if task in ["sst2", "cola"]:
#         return dataset
#     elif task in ['mrpc', "rte"]:
#         def format_data(example):
#             example['sentence'] = example['sentence1'] + tokenizer.sep_token + example['sentence2'] 
#             return example
#         dataset = dataset.map(format_data)
#         dataset = dataset.remove_columns(["sentence1", 'sentence2'])
#         return dataset
#     elif task=="qnli":
#         def format_qnli(example):
#             example['sentence'] = example['question'] + tokenizer.sep_token + example['sentence']
#             return example
#         dataset = dataset.map(format_qnli)
#         dataset = dataset.remove_columns(["question"])
#         return dataset
#     elif task=="stsb":
#         return None
#     else:
#         raise ValueError(f"Task {task} not supported")
        
# train_dataset = get_dataset("glue", task, "train")
# eval_dataset = get_dataset("glue", task, "validation")
# test_dataset = get_dataset("glue", task, "test")
#metric = evaluate.load("glue", task)

# def tokenize_function(examples):
#     outputs = tokenizer(examples["sentence"], truncation=True, padding=True)
#     return outputs

# tokenized_dataset = datasets.map(
#     tokenize_function,
#     batched=True,
# )

# from transformers import DataCollatorWithPadding
# collate_fn = DataCollatorWithPadding(tokenizer, max_length=max_length)
# train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
# eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn, batch_size=batch_size)

def tokenize_function(examples):
    # max_length=None => use the model max length (it's actually the default)
    outputs = tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_length)
    return outputs


tokenized_datasets = datasets.map(
    tokenize_function,
    batched=True,
    remove_columns=["idx", "sentence1", "sentence2"],
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

model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, return_dict=True, max_length=None)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
model

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
    predictions = outputs.logits.argmax(dim=-1)
    predictions, references = predictions, batch["labels"]
    metric.add_batch(
        predictions=predictions,
        references=references,
    )
#print("testtt:", test_dataloader)

eval_metric = metric.compute()
print(eval_metric)