from albumentations import *


def train_aug(image_size=224):
    return Compose([
        RandomCrop(image_size, image_size),
        Flip(),
        RandomRotate90(),
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        Resize(image_size, image_size),
    ], p=1)
