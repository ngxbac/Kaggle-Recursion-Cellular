#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
RUN_CONFIG=config_control.yml

#[1,2,4,5,6] [1,3,4,5,6] [2,3,4,5,6]

for channels in [1,2,3,4,5]; do
    LOGDIR=/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/$channels/se_resnext101_32x4d/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --stages/data_params/channels=$channels:list \
        --verbose
done