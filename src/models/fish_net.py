from .fishnet import fishnet99, fishnet150
from cnn_finetune import make_model
from .utils import *


class Fishnet(nn.Module):
    def __init__(self,  model_name="fishnet99",
                        pretrained=None,
                        num_classes=1108,
                        n_channels=6):
        super(Fishnet, self).__init__()

        if model_name == 'fishnet99':
            self.model = fishnet99(
                pretrained=pretrained,
                n_class=num_classes
            )
            self.fc = self.model.fish.fish[9][4][1]
        elif model_name == 'fishnet150':
            self.model = fishnet150(
                pretrained=pretrained,
                n_class=num_classes
            )
            self.fc = self.model.fish.fish[9][4][1]
        else:
            raise Exception("Invalid model name !")

        conv1 = self.model.conv1[0]
        self.model.conv1[0] = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model.conv1[0].weight.data[:, :3, :, :] = conv1.weight.data
        self.model.conv1[0].weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

    def forward(self, x):
        out, score_feat = self.model(x)
        return out

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.fc.parameters():
            param.requires_grad = True
