import torch.nn as nn
import pretrainedmodels
from cnn_finetune import make_model


def cell_senet(model_name='se_resnext50', num_classes=1108, n_channels=6):
    # model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model = make_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True
    )
    conv1 = model.layer0.conv1
    model.layer0.conv1 = nn.Conv2d(in_channels=n_channels,
                                   out_channels=conv1.out_channels,
                                   kernel_size=conv1.kernel_size,
                                   stride=conv1.stride,
                                   padding=conv1.padding,
                                   bias=conv1.bias)

    # copy pretrained weights
    model.layer0.conv1.weight.data[:, :3, :, :] = conv1.weight.data
    model.layer0.conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels-3), :, :]

    model.avgpool = nn.AdaptiveAvgPool2d(1)
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, num_classes)
    return model
