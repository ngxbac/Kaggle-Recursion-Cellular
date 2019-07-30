import torch.nn as nn
import pretrainedmodels
import torch
from torchvision import models
from cnn_finetune import make_model
import timm
from .utils import *


class ResNet(nn.Module):
    def __init__(self,  model_name="resnet50",
                        num_classes=1108,
                        n_channels=6):
        super(ResNet, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class ResNet50CutMix(nn.Module):
    def __init__(self,  num_classes=1108,
                        pretrained=None,
                        n_channels=6):
        super(ResNet50CutMix, self).__init__()

        self.model = models.resnet50(pretrained=False)
        if pretrained:
            checkpoint = torch.load(pretrained)['model']

            model_dict = self.model.state_dict()
            for k in model_dict.keys():
                if (('module.' + k) in checkpoint.keys()):
                    model_dict[k] = checkpoint.get(('module.' + k))
                else:
                    print("{} is not in dict !".format(k))

            self.model.load_state_dict(model_dict)
            print("Loaded checkpoint: ", pretrained)

        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

        dim_feats = self.model.fc.in_features
        self.model.fc = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True