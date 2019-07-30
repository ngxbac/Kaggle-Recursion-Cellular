#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config.yml


for channels in [1,2,3,4,5]; do
    for fold in 0; do
        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/test/190730/cutmix/fold_$fold/resnet50/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=./csv/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/valid_$fold.csv:str \
            --verbose
    done
done