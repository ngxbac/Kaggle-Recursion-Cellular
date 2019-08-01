import torch
import torch.nn as nn
from .fishnet import fish


def fishnet150(pretrained, n_class, **kwargs):
    """
    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14 | 7,   14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7 | 14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 4, 8, 4, 2, 2, 2, 2, 2, 4],
        'num_trans_blks': [2, 2, 2, 2, 2, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}
    model = fish(**cfg)

    if pretrained:
        state_dict = torch.load(pretrained)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove module prefix
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)

    model.fish.fish[9][4][1] = nn.Conv2d(1056, n_class, kernel_size=(1, 1), stride=(1, 1))

    return model


def fishnet99(pretrained, n_class, **kwargs):
    """
    :return:
    """
    net_cfg = {
        #  input size:   [224, 56, 28,  14 | 7,   14,  28 | 56,   28,  14]
        # output size:   [56,  28, 14,   7 | 14,  28,  56 | 28,   14,   7]
        #                  |    |    |   |    |    |    |    |     |    |
        'network_planes': [64, 128, 256, 512, 512, 512, 384, 256, 320, 832, 1600],
        'num_res_blks': [2, 2, 6, 2, 1, 1, 1, 1, 2, 2],
        'num_trans_blks': [1, 1, 1, 1, 1, 4],
        'num_cls': 1000,
        'num_down_sample': 3,
        'num_up_sample': 3,
    }
    cfg = {**net_cfg, **kwargs}

    model = fish(**cfg)

    if pretrained:
        state_dict = torch.load(pretrained)['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove module prefix
            k = k.replace('module.', '')
            new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=True)

    model.fish.fish[9][4][1] = nn.Conv2d(1056, n_class, kernel_size=(1, 1), stride=(1, 1))

    return model