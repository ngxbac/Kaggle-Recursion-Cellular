from albumentations import *
import albumentations.augmentations.functional as F
import random


class ChannelDropoutCustom(ImageOnlyTransform):
    """Randomly Drop Channels in the input Image.

    Args:
        channel_drop_range (int, int): range from which we choose the number of channels to drop.
        fill_value : pixel value for the dropped channel.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, uint16, unit32, float32
    """

    def __init__(self, channel_drop_range=(1, 1), fill_value=0, always_apply=False, p=0.5):
        super(ChannelDropoutCustom, self).__init__(always_apply, p)

        self.min_channels = channel_drop_range[0]
        self.max_channels = channel_drop_range[1]

        assert 1 <= self.min_channels <= self.max_channels

        self.fill_value = fill_value

    def apply(self, img, channels_to_drop=(0, ), **params):
        return F.channel_dropout(img, channels_to_drop, self.fill_value)

    def get_params_dependent_on_targets(self, params):
        img = params['image']

        num_channels = img.shape[-1]

        if len(img.shape) == 2 or num_channels == 1:
            raise NotImplementedError("Images has one channel. ChannelDropout is not defined.")

        if self.max_channels >= num_channels:
            raise ValueError("Can not drop all channels in ChannelDropout.")

        num_drop_channels = random.randint(self.min_channels, self.max_channels)

        drop_channels = list(set(range(num_channels)) - set([0]))

        channels_to_drop = random.choice(drop_channels, size=num_drop_channels, replace=False)

        return {'channels_to_drop': channels_to_drop}

    def get_transform_init_args_names(self):
        return ('channel_drop_range', 'fill_value')


def train_aug(image_size=224):
    return Compose([
        ChannelDropoutCustom(),
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