#!/bin/bash

env_name="locvqallm"
dataset="insegcat"
annotation="./data/insegcat/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/INSEGCAT_v1/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/insegcat/v1_miccai2023/checkpoints/checkpoint_epoch11_step21114_bleu0.000959_cider4.470721_acc0.894144.pth"

version="v1_miccai2023"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --test \
    --miccai2023 \
    --baseline "miccai2023" \
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