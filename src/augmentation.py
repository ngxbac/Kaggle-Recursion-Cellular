from albumentations import *


def train_aug(image_size=224):
    return Compose([
        # RandomCrop(448, 448),
        Resize(image_size, image_size),
        RandomRotate90(),
        Flip(),
        Transpose(),
        # HorizontalFlip(),
        # Normalize(),
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        # CenterCrop(448, 448),
        Resize(image_size, image_size)
        # Normalize(),
    ], p=1)
