#!/bin/bash

env_name="locvqallm"
dataset="dme"
annotation="./data/dme/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/DME_v2/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/dme/v1_ours3/checkpoints/checkpoint_epoch10_step6415_bleu0.000948_cider4.325000_acc0.865000.pth"

version="v1_ours3"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --validate \
    --ours \
    --ours_version "ours3" \
    --baseline "context_only" \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 8 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --llm_freeze False \
    --savedmodel_path ${savepath} \
    --max_length 4 \
    --min_new_tokens 1 \
    --max_new_tokens 15 \
    --repetition_penalty 2.0 \
    --length_penalty -1.0 \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt