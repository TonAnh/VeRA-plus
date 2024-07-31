#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python main_lora.py \
    --batch_size 16 \
    --model_name_or_path "roberta-base" \
    --task "sst2" \
    --num_epochs 60 \
    --max_length 512 \
    --r 8 \
    --lora_alpha 8 \
    --lr 5e-4