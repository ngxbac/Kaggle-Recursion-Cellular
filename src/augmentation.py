from albumentations import *

import itertools


def train_aug(image_size=224):
    # policies = './csv/best_policy.data'
    # with open(policies, 'r') as fid:
    #     policies = eval(fid.read())
    #     policies = itertools.chain.from_iterable(policies)

    # aug_list = []
    # for policy in policies:
    #     op_1, params_1 = policy[0]
    #     op_2, params_2 = policy[1]
    #     aug = Compose([
    #         globals().get(op_1)(**params_1),
    #         globals().get(op_2)(**params_2),
    #     ])
    #     aug_list.append(aug)

    return Compose([
        ChannelDropout(),
        # CLAHE(),
        RandomRotate90(),
        Flip(),
        Transpose(),
        IAAAffine(shear=(-10, 10)),
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


def test_tta(image_size):
    test_dict = {
        'normal': Compose([
            Resize(image_size, image_size)
        ]),
        # 'hflip': Compose([
        #     HorizontalFlip(p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
        # 'rot90': Compose([
        #     Rotate(limit=(90, 90), p=1),
        #     Resize(image_size, image_size),
        # ], p=1),
    }

    return test_dict