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
where `model_name_or_path` can be "roberta-base" or "roberta-large".
For `task`, we can choose between "sst2", "mrpc", "cola", "qnli", "rte", or "stsb".
Set `use_rsvera` "true" if we want to apply **rank stabilization** or "false" otherwise. Note that adjust the values of other configuration parameters corresponding to the task and model we want to run according to the reference table below.
![image](https://github.com/lyntrann/VeRA-plus/assets/90293410/543bb5a2-99b7-4bbc-8edb-387d9426d51c)
We adjust the configurations for VeRA and VeRA-plus similarly according to table 2 above.
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

