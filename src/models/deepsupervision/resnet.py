import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from catalyst.contrib.modules.common import Flatten
from catalyst.contrib.modules.pooling import GlobalConcatPool2d
from cnn_finetune import make_model


class DSResnet(nn.Module):
    def __init__(
        self,
        model_name='resnet50',
        num_classes=6,
        pretrained=True,
        n_channels=4,

    ):
        super(DSResnet, self).__init__()
        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout_p=0.3
        )
        # print(self.model)
        conv1 = self.model._features[0]
        self.model._features[0] = nn.Conv2d(in_channels=n_channels,
                                out_channels=conv1.out_channels,
                                kernel_size=conv1.kernel_size,
                                stride=conv1.stride,
                                padding=conv1.padding,
                                bias=conv1.bias)

        # copy pretrained weights
        self.model._features[0].weight.data[:,:3,:,:] = conv1.weight.data
        self.model._features[0].weight.data[:,3:n_channels,:,:] = conv1.weight.data[:,:int(n_channels-3),:,:]

        # self.deepsuper_1 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(),
        #     Flatten(),
        #     nn.BatchNorm1d(256),
        #     nn.Linear(256, num_classes)
        # )

        self.deepsuper_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes)
        )

        self.deepsuper_3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        )

        self.is_infer = False

    def freeze_base(self):
        # pass
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        # pass
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # block 0
        x = self.model._features[0](x)
        x = self.model._features[1](x)
        x = self.model._features[2](x)
        x = self.model._features[3](x)
        # x_1 = self.deepsuper_1(x)

        # block 1
        x = self.model._features[4](x)
        # block 2
        x = self.model._features[5](x)
        x_2 = self.deepsuper_2(x)
        # block 3
        x = self.model._features[6](x)
        x_3 = self.deepsuper_3(x)
        # block 4
        x = self.model._features[7](x)
        x = self.model.pool(x)
        x = x.view(x.size(0), -1)
        x_final = self.model._classifier(x)

        return x_2, x_3, x_final

    def freeze(self):
        # Freeze all the backbone
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        # Unfreeze all the backbone
        for param in self.model.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    x = torch.zeros((2, 4, 512, 512))
    model = DSResnet()
    out = model(x)