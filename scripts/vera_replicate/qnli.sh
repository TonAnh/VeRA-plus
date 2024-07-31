#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python rsvera.py \
    --batch_size 64 \
    --model_name_or_path "roberta-base" \
    --task "qnli" \
    --num_epochs 25 \
    --max_length 512 \
    --r 1024 \
    --vera_alpha 8 \
    --use_rsvera \
    --head_lr 4e-3 \
    --vera_lr 1e-2