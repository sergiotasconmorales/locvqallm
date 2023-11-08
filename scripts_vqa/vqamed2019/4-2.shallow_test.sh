#!/bin/bash

env_name="locvqallm"
dataset="vqamed2019"
annotation="./data/vqa_med_2019/processed/anns.json"
base_dir="./data/vqa_med_2019/original"
delta_file="/storage/homefs/st20f757/vqa/ReportWizard/save/mimic_cxr/v1_shallow/checkpoints/checkpoint_epoch4_step169240_bleu0.170310_cider0.277632.pth"

version="v1_shallow"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --test \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 4 \
    --freeze_vm True \
    --vis_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 100 \
    --min_new_tokens 80 \
    --max_new_tokens 120 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt