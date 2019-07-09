#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config.yml


for channels in [1,2,3,5] [1,2,3,6] [1,2,4,5] [1,2,4,6] [1,2,5,6] [1,3,4,5] [1,3,4,6] [1,3,5,6] [1,4,5,6] [2,3,4,5] [2,3,4,6] [2,3,5,6] [2,4,5,6] [3,4,5,6]; do
    LOGDIR=/raid/bac/kaggle/logs/recursion_cell/search_channels/$channels/se_resnext50_32x4d/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --stages/data_params/channels=$channels:list \
        --verbose
done