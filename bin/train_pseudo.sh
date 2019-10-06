#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1
RUN_CONFIG=config_pseudo.yml


PRETRAINED_CONTROL=/logs/pretrained_controls/
model=se_resnext50_32x4d
for channels in [1,2,3,4,5] [1,2,3,4,6] [1,2,3,5,6] [1,2,4,5,6] [1,3,4,5,6] [2,3,4,5,6]; do
    for fold in 0 1 2 3 4; do 
        TRAIN_CSV=./csv/pseudo/train_$fold.csv
        VALID_CSV=./csv/pseudo/valid_$fold.csv
        LOGDIR=/logs/pseudo/$channels/fold_$fold/$model/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=$TRAIN_CSV:str \
            --stages/data_params/valid_csv=$VALID_CSV:str \
            --model_params/weight=$PRETRAINED_CONTROL/$channels/$model/checkpoints/best.pth:str \
            --verbose
    done
done