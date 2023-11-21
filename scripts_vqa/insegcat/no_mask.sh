#!/bin/bash

env_name="locvqallm"
dataset="insegcat"
annotation="./data/insegcat/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/INSEGCAT_v1/images"

version="v1_insegcat"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --baseline "no_mask" \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 12\
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --llm_freeze False \
    --llama_model "meta-llama/Llama-2-7b-hf" \
    --precision 32 \
    --savedmodel_path ${savepath} \
    --learning_rate 1e-4 \
    --gradient_clip_val 1 \
    --max_length 4 \
    --repetition_penalty 2.0 \
    --length_penalty -1.0 \
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
