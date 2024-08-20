#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
export WORKDIR=./
export PYTHONPATH="$WORKDIR:$PYTHONPATH"
python vera-smoe.py \
    --batch_size 64 \
    --output_dir "output/vera-smoe" \
    --model_name_or_path roberta-base \
    --task mrpc \
    --num_epochs 30 \
    --max_length 512 \
    --r 1024 \
    --vera_alpha 8 \
    --use_rsvera True \
    --head_lr 4e-3 \
    --vera_lr 1e-2 \
    --num_experts 4 \
    --top_k 1 \