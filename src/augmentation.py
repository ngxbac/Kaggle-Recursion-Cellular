from albumentations import *

import itertools


def train_aug(image_size=224):
    policies = './csv/best_policy.data'
    with open(policies, 'r') as fid:
        policies = eval(fid.read())
        policies = itertools.chain.from_iterable(policies)

    aug_list = []
    for policy in policies:
        op_1, params_1 = policy[0]
        op_2, params_2 = policy[1]
        aug = Compose([
            globals().get(op_1)(**params_1),
            globals().get(op_2)(**params_2),
        ])
        aug_list.append(aug)

    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        # OneOf(
        #     aug_list
        # ),
        Resize(image_size, image_size)
    ], p=1)


def valid_aug(image_size=224):
    return Compose([
        # CenterCrop(448, 448),
        Resize(image_size, image_size)
        # Normalize(),
    ], p=1)
