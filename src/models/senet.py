import torch
import torch.nn as nn
import pretrainedmodels
from cnn_finetune import make_model
from .utils import *


class SENet(nn.Module):
    def __init__(self,  model_name="se_resnext50_32x4d",
                        num_classes=1108,
                        n_channels=6):
        super(SENet, self).__init__()

        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            # pool=GlobalConcatPool2d(),
            # classifier_factory=make_classifier
        )
        self.conv = Conv2dSame(
            in_channels=n_channels,
            out_channels=3,
            kernel_size=1,
            stride=1,
        )

    def forward(self, x):
        x = self.conv(x)
        return self.model(x)


def cell_senet(model_name='se_resnext50_32x4d', num_classes=1108, n_channels=6):
    model = make_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True,
        # pool=GlobalConcatPool2d(),
        # classifier_factory=make_classifier
    )
    # print(model)
    conv1 = model._features[0].conv1
    model._features[0].conv1 = nn.Conv2d(in_channels=n_channels,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # copy pretrained weights
    model._features[0].conv1.weight.data[:,:3,:,:] = conv1.weight.data
    model._features[0].conv1.weight.data[:,3:n_channels,:,:] = conv1.weight.data[:,:int(n_channels-3),:,:]
    # model = SENet(
    #     model_name=model_name,
    #     num_classes=num_classes,
    #     n_channels=n_channels
    # )
    #
    # for param in model.parameters():
    #     param.requires_grad = True
    return model


def hpa_cell_senet(model_name='se_resnext50_32x4d', num_classes=1108, n_channels=6, pretrained=None):
    model = cell_senet(model_name=model_name, num_classes=28, n_channels=n_channels)
    if pretrained:
        checkpoint = torch.load(pretrained)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded checkpoint: ", pretrained)

    in_features = model._classifier.in_features
    model._classifier = nn.Linear(
        in_features=in_features,
        out_features=num_classes
    )
    return model
