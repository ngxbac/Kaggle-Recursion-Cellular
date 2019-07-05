from catalyst.contrib.modules.pooling import GlobalConcatPool2d
from catalyst.contrib.modules.common import Flatten
import torch.nn as nn


def make_classifier(in_features, num_classes):
    return nn.Sequential(
        Flatten(),
        nn.BatchNorm1d(in_features * 2),
        nn.Dropout(0.5),
        nn.Linear(in_features * 2, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),
    )
