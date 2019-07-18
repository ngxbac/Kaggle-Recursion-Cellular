from cnn_finetune import make_model
import timm
from .utils import *


class SENetTIMM(nn.Module):
    def __init__(self,  model_name="seresnext26_32x4d",
                        num_classes=1108,
                        n_channels=6):
        super(SENetTIMM, self).__init__()

        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        conv1 = self.model.layer0.conv1
        self.model.layer0.conv1 = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model.layer0.conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model.layer0.conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

    def forward(self, x):
        return self.model(x)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.get_classifier().parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.model.get_classifier().parameters():
            param.requires_grad = True


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
