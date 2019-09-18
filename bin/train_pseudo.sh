#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config_pseudo.yml

#export MASTER_PORT=9669
#export MASTER_ADDR="127.0.0.1"
#export WORLD_SIZE=2
#export RANK=0


PRETRAINED_CONTROL=/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/
model=se_resnext50_32x4d
for channels in [1,2,3,4,5]; do
    for fold in 0 1 2; do
        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/pseudoall2_from_control/$channels/fold_$fold/$model/
        catalyst-dl run \
            --config=./configs/${RUN_CONFIG} \
            --logdir=$LOGDIR \
            --out_dir=$LOGDIR:str \
            --stages/data_params/channels=$channels:list \
            --stages/data_params/train_csv=./csv/pseudo_all2/train_$fold.csv:str \
            --stages/data_params/valid_csv=./csv/pseudo_all2/valid_$fold.csv:str \
            --model_params/weight=$PRETRAINED_CONTROL/$channels/$model/checkpoints/best.pth:str \
            --verbose

    done
done


#PRETRAINED_CONTROL=/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/
#model=se_resnext50_32x4d
#for channels in [1,2,3,4,5]; do
#    for fold in 0; do
#        LOGDIR=/raid/bac/kaggle/logs/recursion_cell/pseudo_from_control_sync/$channels/fold_$fold/$model/
#        RANK=0 LOCAL_RANK=0 catalyst-dl run \
#            --config=./configs/${RUN_CONFIG} \
#            --logdir=$LOGDIR \
#            --out_dir=$LOGDIR:str \
#            --stages/data_params/channels=$channels:list \
#            --stages/data_params/train_csv=./csv/pseudo/train_$fold.csv:str \
#            --stages/data_params/valid_csv=./csv/pseudo/valid_$fold.csv:str \
#            --model_params/weight=$PRETRAINED_CONTROL/$channels/$model/checkpoints/best.pth:str \
#            --verbose \
#            --distributed_params/rank=0:int
#
#        sleep 5
#
#        RANK=1 LOCAL_RANK=1 catalyst-dl run \
#            --config=./configs/${RUN_CONFIG} \
#            --logdir=$LOGDIR \
#            --out_dir=$LOGDIR:str \
#            --stages/data_params/channels=$channels:list \
#            --stages/data_params/train_csv=./csv/pseudo/train_$fold.csv:str \
#            --stages/data_params/valid_csv=./csv/pseudo/valid_$fold.csv:str \
#            --model_params/weight=$PRETRAINED_CONTROL/$channels/$model/checkpoints/best.pth:str \
#            --verbose \
#            --distributed_params/rank=1:int
#    done
#done