from cnn_finetune import make_model
import timm
from .utils import *


class InceptionV3TIMM(nn.Module):
    def __init__(self,  model_name="gluon_inception_v3",
                        num_classes=1108,
                        n_channels=6):
        super(InceptionV3TIMM, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        conv1 = self.model.Conv2d_1a_3x3.conv
        self.model.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model.Conv2d_1a_3x3.conv.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.Conv2d_1a_3x3.conv.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True