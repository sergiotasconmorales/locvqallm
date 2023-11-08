#!/bin/bash

env_name="locvqallm"
dataset="vqamed2019"
annotation="./data/vqa_med_2019/processed/anns.json"
base_dir="./data/vqa_med_2019/original"

version="v1_shallow"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa True \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 4 \
    --val_batch_size 4\
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --learning_rate 1e-4 \
    --gradient_clip_val 1 \
    --max_length 100 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 4 \
    --devices 2 \
    --max_epochs 30 \
    --limit_val_batches 0.5 \
    --val_check_interval 0.5 \
    --num_sanity_val_steps 2 \
    --strategy "ddp"\
    --low_resource False \
    --min_new_tokens 1 \
    --max_new_tokens 15 \
    
    2>&1 |tee -a ${savepath}/log.txt
