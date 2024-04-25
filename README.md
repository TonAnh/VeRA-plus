# VeRA-plus

## Table of Contents
- [Installation](#installation)
- [Example](#example)

## Installation
To install the required packages in a new environment, follow these steps:
```python
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install peft
pip install evaluate
```
## Example 
### LoRA
```python
python main_lora.py \
--batch_size 16 \
--model_name_or_path "roberta-base" \
--task "mrpc" \
--num_epochs 30 \
--max_length 512 \
--r 8 \
--lora_alpha 8 \
--use_rslora true \
--lr 4e-4
```
For `**model_name_or_path**`, users can choose between "roberta-base" or "roberta-large". They should replace the placeholder text with their desired option.
For task, users can choose between "sst2", "mrpc", "cola", "qnli", "rte", or "stsb". Again, they should replace the placeholder text with their desired option.
For use_rsvera, users can set it to "true" if they want to apply rank stabilization or "false" otherwise. They should replace the placeholder text accordingly.
### VeRA
```python
python rsvera.py \
--batch_size 64 \
--model_name_or_path "roberta-base" \
--task "mrpc" \
--num_epochs 30 \
--max_length 512 \
--r 1024 \
--vera_alpha 8 \
--use_rsvera true \
--head_lr 4e-3 \
--vera_lr 1e-2
```
### VeRA-plus
```python
python rsvera.py\
 --batch_size 64\
 --model_name_or_path roberta-base\
 --task mrpc\
 --num_epochs 30\
 --max_length 512\
 --r 1024\
 --vera_alpha 8\
 --use_rsvera true\
 --head_lr 4e-3\
 --vera_lr 1e-2\
```

