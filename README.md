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
--use_rslora "true" \
--lr 4e-4
```

