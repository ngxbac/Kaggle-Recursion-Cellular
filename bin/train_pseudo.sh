#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config_pseudo.yml


PRETRAINED_CONTROL=/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/
for channels in [1,2,3,4,6] [1,2,3,5,6]; do
    for fold in 0 1 2 3 4; do
        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/pseudo_from_control/$channels/fold_$fold/se_resnext50_32x4d/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=./csv/pseudo/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/pseudo/valid_$fold.csv:str \
            --model_params/weight=$PRETRAINED_CONTROL/$channels/se_resnext50_32x4d/checkpoints/best.pth:str \
            --verbose
    done
done