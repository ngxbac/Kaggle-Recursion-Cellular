#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1,2
RUN_CONFIG=config.yml


LOGDIR=/raid/bac/kaggle/logs/recursion_cell/test/c1234_s1_affine_warmup_optimal_channel/se_resnex50_32x4d/
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --verbose