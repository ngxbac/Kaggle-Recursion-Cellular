import torch.nn as nn
import pretrainedmodels
from cnn_finetune import make_model


def cell_resnet(model_name, num_classes=1108, n_channels=6):
    model = make_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=True
    )
    conv1 = model._features[0]
    model._features[0] = nn.Conv2d(in_channels=n_channels,
                            out_channels=conv1.out_channels,
                            kernel_size=conv1.kernel_size,
                            stride=conv1.stride,
                            padding=conv1.padding,
                            bias=conv1.bias)

    # copy pretrained weights
    model._features[0].weight.data[:,:3,:,:] = conv1.weight.data
    model._features[0].weight.data[:,3:n_channels,:,:] = conv1.weight.data[:,:int(n_channels-3),:,:]
    return model


if __name__ == '__main__':
    import torch
    model = cell_resnet(model_name='resnet18')
    x = torch.randn((1, 6, 320, 320))
    y = model(x)