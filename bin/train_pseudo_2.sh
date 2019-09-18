#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
RUN_CONFIG=config_pseudo.yml


PRETRAINED_CONTROL=/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/
model=se_resnext50_32x4d
for channels in [1,2,3,4,5]; do
    for fold in 3 4; do
        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/pseudoall_from_control/$channels/fold_$fold/$model/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=./csv/pseudo_all/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/pseudo_all/valid_$fold.csv:str \
            --model_params/weight=$PRETRAINED_CONTROL/$channels/$model/checkpoints/best.pth:str \
            --verbose

    done
done