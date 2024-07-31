#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python main_lora.py \
    --batch_size 32 \
    --model_name_or_path "roberta-base" \
    --task "cola" \
    --num_epochs 80 \
    --max_length 512 \
    --r 8 \
    --lora_alpha 8 \
    --use_rslora \
    --lr 4e-4