#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python rsvera.py \
    --batch_size 64 \
    --model_name_or_path "roberta-base" \
    --task "sst2" \
    --num_epochs 60 \
    --max_length 512 \
    --r 1024 \
    --vera_alpha 8 \
    --use_rsvera \
    --head_lr 4e-3 \
    --vera_lr 4e-3