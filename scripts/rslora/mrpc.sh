#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python main_lora.py \
    --batch_size 16 \
    --model_name_or_path "roberta-base" \
    --task "mrpc" \
    --num_epochs 30 \
    --max_length 512 \
    --r 8 \
    --lora_alpha 8 \
    --use_rslora \
    --lr 4e-4