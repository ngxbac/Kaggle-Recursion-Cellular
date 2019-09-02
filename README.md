# Requirements

- torch == 1.1.0
- cnn_finetune == 0.5.3
- albumentations == 0.2.3
- catalyst == 19.06.5

# How to train 
## Pretrained with controls
```bash
bash bin/train_control.sh
```

Pretrained models are saved at:
`/raid/bac/kaggle/logs/recursion_cell/pretrained_controls/$channels/se_resnext50_32x4d/`  
where `channels` can be: `[1,2,3,4,5], etc`

## Train with pseudo data
```bash
bash bin/train_pseudo.sh
```

* `PRETRAINED_CONTROL`: is the root folder of pretrained models above  

* `--model_params/weight=
$PRETRAINED_CONTROL/$channels/se_resnext50_32x4d/checkpoints/best.pth:str \`  is the weight of 
corresponding model pretrained on controls dataset.


## Train with usually data
Similar to `Train with pseudo data` part
```bash
bash bin/train.sh
```

* `PRETRAINED_CONTROL`: is the root folder of pretrained models above  

* `--model_params/weight=
$PRETRAINED_CONTROL/$channels/se_resnext50_32x4d/checkpoints/best.pth:str \`  is the weight of 
corresponding model pretrained on controls dataset.



- Run: `python src/make_submission.csv`