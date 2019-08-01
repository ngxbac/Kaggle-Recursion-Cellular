#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=3
RUN_CONFIG=config_ds.yml


for channels in [1,2,3,4,5,6]; do
    for fold in 0; do
        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/test/190731/fold_$fold/DSInceptionV3/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=./csv/kfold5/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/kfold5/valid_$fold.csv:str \
            --verbose
    done
done