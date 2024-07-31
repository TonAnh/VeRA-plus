#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python rsvera.py \
    --batch_size 64 \
    --model_name_or_path "roberta-base" \
    --task "cola" \
    --num_epochs 80 \
    --max_length 512 \
    --r 1024 \
    --vera_alpha 8 \
    --use_rsvera \
    --head_lr 1e-2 \
    --vera_lr 1e-2