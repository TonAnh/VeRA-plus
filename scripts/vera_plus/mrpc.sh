#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python vera-plus.py \
    --batch_size 64 \
    --model_name_or_path roberta-base \
    --task mrpc \
    --num_epochs 30 \
    --max_length 512 \
    --r 1024 \
    --vera_alpha 8 \
    --use_rsvera True \
    --head_lr 4e-3\
    --vera_lr 1e-2\