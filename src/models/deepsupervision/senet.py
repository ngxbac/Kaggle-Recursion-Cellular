import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from catalyst.contrib.modules.common import Flatten
from catalyst.contrib.modules.pooling import GlobalConcatPool2d
from cnn_finetune import make_model


class DSSENet(nn.Module):
    def __init__(
        self,
        model_name='se_resnext50_32x4d',
        num_classes=6,
        pretrained=True,
        n_channels=4,

    ):
        super(DSSENet, self).__init__()
        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout_p=0.3
        )
        # print(self.model)
        conv1 = self.model._features[0].conv1
        self.model._features[0].conv1 = nn.Conv2d(in_channels=n_channels,
                                out_channels=conv1.out_channels,
                                kernel_size=conv1.kernel_size,
                                stride=conv1.stride,
                                padding=conv1.padding,
                                bias=conv1.bias)

        # copy pretrained weights
        self.model._features[0].conv1.weight.data[:,:3,:,:] = conv1.weight.data
        self.model._features[0].conv1.weight.data[:,3:n_channels,:,:] = conv1.weight.data[:,:int(n_channels-3),:,:]

        self.deepsuper_1 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(256 * 2),
            nn.Linear(256 * 2, num_classes)
        )

        self.deepsuper_2 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(512 * 2),
            nn.Linear(512 * 2, num_classes)
        )

        self.deepsuper_3 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(1024 * 2),
            nn.Linear(1024 * 2, num_classes)
        )

        # WARNING: should adapt the Linear layer to be suitable for each image size !!!
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32768, 1024), # Take care here: 3200 for 224x224, 25088 for 512x512
            nn.ReLU(),
            nn.Dropout(0.3),
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
        x = self.model._features[0](x)
        x = self.model._features[1](x)
        x_1 = self.deepsuper_1(x)
        x = self.model._features[2](x)
        x_2 = self.deepsuper_2(x)
        x = self.model._features[3](x)
        x_3 = self.deepsuper_3(x)
        x = self.model._features[4](x)
        x_final = self.fc(x)

        return x_1, x_2, x_3, x_final

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
    model = DSSENet()
    out = model(x)