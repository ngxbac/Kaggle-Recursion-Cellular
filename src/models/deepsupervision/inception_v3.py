import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from catalyst.contrib.modules.common import Flatten
from catalyst.contrib.modules.pooling import GlobalConcatPool2d


class DSInceptionV3(nn.Module):
    def __init__(
        self,
        num_classes=6,
        pretrained=True,
        n_channels=4,

    ):
        super(DSInceptionV3, self).__init__()
        self.model = models.inception_v3(
            pretrained=pretrained,
            transform_input=False,
            # aux_logits=False
        )

        # Adapt number of channels
        conv1 = self.model.Conv2d_1a_3x3.conv
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        # self.model.Conv2d_1a_3x3.conv.weight.data[:, :3, :, :] = conv1.weight.data
        # self.model.Conv2d_1a_3x3.conv.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

        self.deepsuper_2 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(288 * 2),
            nn.Linear(288 * 2, num_classes)
        )

        self.deepsuper_4 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(768 * 2),
            nn.Linear(768 * 2, num_classes)
        )

        self.deepsuper_6 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(768 * 2),
            nn.Linear(768 * 2, num_classes)
        )

        self.deepsuper_8 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(1280 * 2),
            nn.Linear(1280 * 2, num_classes)
        )

        self.deepsuper_9 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(2048 * 2),
            nn.Linear(2048 * 2, num_classes)
        )

        self.deepsuper_10 = nn.Sequential(
            GlobalConcatPool2d(),
            Flatten(),
            nn.BatchNorm1d(2048 * 2),
            nn.Linear(2048 * 2, num_classes)
        )

        # WARNING: should adapt the Linear layer to be suitable for each image size !!!
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=128, kernel_size=(1, 1)),
            nn.ReLU(),
            Flatten(),
            nn.Linear(25088, 1024), # Take care here: 3200 for 224x224, 25088 for 512x512
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
        if self.model.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)  # => Finish first convs

        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)  # => Finish mixed 0
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)  # => Finish mixed 1
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)  # => Finish mixed 2
        # import pdb
        # pdb.set_trace()
        x_mix_2 = self.deepsuper_2(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x) # => Finish mixed 3
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)  # => Finish mixed 4
        x_mix_4 = self.deepsuper_4(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)  # => Finish mixed 5
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)  # => Finish mixed 6
        x_mix_6 = self.deepsuper_6(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)  # => Finish mixed 7
        # 17 x 17 x 768
        # if self.model.training and self.model.aux_logits:
        #     aux = self.model.AuxLogits(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)   # => Finish mixed 8
        x_mix_8 = self.deepsuper_8(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)   # => Finish mixed 9
        x_mix_9 = self.deepsuper_9(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)   # => Finish mixed 10
        # 8 x 8 x 2048

        # here is the model output
        x_mix_10 = self.deepsuper_10(x)
        x_final = self.fc(x)

        return x_mix_2, x_mix_4, x_mix_6, x_mix_8, x_mix_9, x_mix_10, x_final

    def freeze(self):
        # Freeze all the backbone
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze(self):
        # Unfreeze all the backbone
        for param in self.model.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    model = DSInceptionV3()