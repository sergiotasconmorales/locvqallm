#!/bin/bash

env_name="locvqallm"
dataset="coco_regions"
annotation="./data/coco_regions/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/COCO-Regions_v1/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/coco_regions/v1_context_only/checkpoints/checkpoint_epoch12_step130500_bleu0.000857_cider3.268101_acc0.653620.pth"

version="v1_context_only"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --test \
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