# Requirements

- torch == 1.1.0
- cnn_finetune == 0.5.3
- albumentations == 0.2.3
- catalyst == 19.06.5

# How to train 
- In `configs/config.yml`  
Change: `stages/data_params/root` to your root data that you download from Kaggle

- In `bin/train.sh`  
Change: 
  - `export CUDA_VISIBLE_DEVICES=2,3` which GPUs are used for training
  - `LOGDIR`: saved checkpoints, logs, etc
  
- Run: `bash bin/train.sh` 
- Take a coffe or go to sleep


# How to predict 
Please refer `make_submission.py`. 

- Change following:
```python
    test_csv = '/raid/data/kaggle/recursion-cellular-image-classification/test.csv'
    # test_csv = './csv/valid_0.csv'
    log_dir = "/raid/bac/kaggle/logs/recursion_cell/test/rgb_no_crop_512_accum2_456/se_resnext50_32x4d/"
    root = "/raid/data/kaggle/recursion-cellular-image-classification/"

```

- Run: `python src/make_submission.csv`