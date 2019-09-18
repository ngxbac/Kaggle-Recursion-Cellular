#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config.yml

# [1,2,3,4,5] [1,2,3,4,6] [1,2,3,5,6]

PRETRAINED_CONTROL=/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/
for channels in [1,2,3,4,5]; do
    for fold in 0 1 2 3 4; do
        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/normal_from_control/$channels/fold_$fold/densenet121/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=./csv/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
            --model_params/weight=$PRETRAINED_CONTROL/$channels/densenet121/checkpoints/best.pth:str \
            --verbose
    done
done