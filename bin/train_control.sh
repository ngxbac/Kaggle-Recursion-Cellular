#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config_control.yml

#[1,2,4,5,6] [1,3,4,5,6] [2,3,4,5,6]

model_name=se_resnext50_32x4d
for channels in [1,2,3,4,5] [1,2,3,4,6] [1,2,3,5,6] [1,2,4,5,6] [1,3,4,5,6] [2,3,4,5,6]; do
    LOGDIR=/logs/pretrained_controls/${channels}/${model_name}/
    catalyst-dl run \
        --config=./configs/${RUN_CONFIG} \
        --logdir=$LOGDIR \
        --out_dir=$LOGDIR:str \
        --stages/data_params/channels=$channels:list \
        --model_params/model_name=${model_name}:str \
        --verbose
done