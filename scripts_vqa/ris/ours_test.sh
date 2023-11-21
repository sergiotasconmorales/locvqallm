#!/bin/bash

env_name="locvqallm"
dataset="ris"
annotation="./data/ris/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/STS2017_v2/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/ris/v4_ours/checkpoints/checkpoint_epoch5_step5837_bleu0.000931_cider4.120935_acc0.824187.pth"

version="v4_ours"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --test \
    --ours \
    --baseline "mask" \
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
    --max_length 100 \
    --min_new_tokens 1 \
    --max_new_tokens 15 \
    --repetition_penalty 2.0 \
    --length_penalty -1.0 \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt