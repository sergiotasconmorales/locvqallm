#!/bin/bash

env_name="locvqallm"
dataset="mimic_cxr"
annotation="data/mimic_cxr/annotation.json"
base_dir="/storage/workspaces/artorg_aimi/ws_00000/sergio/radrep/mimic-cxr-jpg-google/files"
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