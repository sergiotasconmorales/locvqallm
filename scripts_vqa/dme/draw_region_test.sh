#!/bin/bash

env_name="locvqallm"
dataset="dme"
annotation="./data/dme/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/DME_v2/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/dme/v1_draw_region/checkpoints/checkpoint_epoch7_step4887_bleu0.000951_cider4.366667_acc0.873333.pth"

version="v1_draw_region"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --validate \
    --baseline "draw_region" \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 4 \
    --freeze_vm False \
    --precision 32 \
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