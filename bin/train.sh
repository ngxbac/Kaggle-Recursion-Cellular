#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
RUN_CONFIG=config.yml


LOGDIR=./bin/30_epochs/
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --verbose