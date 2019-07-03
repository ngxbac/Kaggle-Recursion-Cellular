from albumentations import *


def train_aug(image_size=224):
    return Compose([
        RandomCrop(image_size, image_size),
        HorizontalFlip(),
        # Normalize(),
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        CenterCrop(image_size, image_size),
        # Normalize(),
    ], p=1)
