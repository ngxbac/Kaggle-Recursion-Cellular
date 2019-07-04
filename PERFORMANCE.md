# Performances

- Baseline  
  - Model: se_resnext50_32x4d
  - image_size: 512x512
  - batch_size: 64
  - grad_accum: 2
  - augmentations:
  ```pythonstub
    def train_aug(image_size=512):
        return Compose([
            Resize(image_size, image_size),
            RandomRotate90(),
            Flip(),
            Transpose(),
        ], p=1)


    def valid_aug(image_size=512):
        return Compose([
            # CenterCrop(448, 448),
            Resize(image_size, image_size)
            # Normalize(),
        ], p=1)
  ```
  
  - Optimizers: 
  ```yaml
  criterion_params:
    criterion: CrossEntropyLoss

  optimizer_params:
    optimizer: Adam
    lr: 0.0003
    weight_decay: 0.0001

  scheduler_params:
    scheduler: MultiStepLR
    milestones: [25, 30, 40]
    gamma: 0.5

  data_params:
    batch_size: 64
    num_workers: 4
    drop_last: False

    image_size: &image_size 512
    train_csv: "./csv/train_0.csv"
    valid_csv: "./csv/valid_0.csv"
    root: "/raid/data/kaggle/recursion-cellular-image-classification/"
    site: 2
    channels: [1, 2, 3]
  ```
  Results: (fold 0)
  
  | Experiment | CV | LB |   
  |:---------|----|---:|  
  |c123_s1| 42.9%| 30.6%|  
  |c123_s2| 41%| 23.6%|  
  |Ensemble 0.7 * c123_s1 + 0.3 * c123_s2 | - | 32.5 |  
  
  c123_s1: using channels=[1,2,3] and site = 1
  