#!/bin/bash

env_name="locvqallm"
dataset="insegcat"
annotation="./data/insegcat/processed/anns.json"
base_dir="/storage/homefs/st20f757/vqa/data/Tools/INSEGCAT_v1/images"
delta_file="/storage/homefs/st20f757/vqa/locvqallm/save/insegcat/v1_no_mask/checkpoints/checkpoint_epoch0_step918_bleu0.000803_cider2.702703_acc0.540541.pth"

version="v1_no_mask"
savepath="./save/$dataset/$version"

~/.conda/envs/${env_name}/bin/python -u train.py \
    --vqa \
    --regions \
    --test \
    --baseline "no_mask" \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --delta_file ${delta_file} \
    --test_batch_size 8 \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --llm_freeze False \
    --llama_model "meta-llama/Llama-2-7b-hf" \
    --savedmodel_path ${savepath} \
    --max_length 4 \
    --min_new_tokens 1 \
    --max_new_tokens 15 \
    --repetition_penalty 2.0 \
    --length_penalty -1.0 \
    --num_workers 12 \
    --devices 1 \
    2>&1 |tee -a ${savepath}/log.txt