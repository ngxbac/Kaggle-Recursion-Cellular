#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config.yml


LOGDIR=/raid/bac/kaggle/logs/recursion_cell/test/c123_s1_1cycle_adamw_norm_per_channel/se_resnext50_32x4d/
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --verbose