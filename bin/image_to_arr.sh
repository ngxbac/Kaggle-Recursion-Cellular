#!/usr/bin/env bash

dataset=train
csv=/raid/data/kaggle/recursion-cellular-image-classification/${dataset}.csv
base_path=/raid/data/kaggle/recursion-cellular-image-classification/
output=/raid/data/kaggle/recursion-cellular-image-classification/array/

python preprocessing/image_to_arr.py image-to-arr   --csv=$csv \
                                                    --base_path=$base_path \
                                                    --output=$output \
                                                    --dataset=$dataset