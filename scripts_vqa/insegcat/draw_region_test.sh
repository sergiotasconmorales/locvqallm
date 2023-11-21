#!/bin/bash

env_name="locvqallm"
dataset="insegcat"
annotation="./data/insegcat/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/INSEGCAT_v1/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/insegcat/v1_draw_region/checkpoints/checkpoint_epoch6_step12852_bleu0.000941_cider4.241742_acc0.848348.pth"

version="v1_draw_region"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --test \
    --baseline "draw_region" \
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