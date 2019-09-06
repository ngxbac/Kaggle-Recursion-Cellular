import torch.nn as nn
import torch
import pretrainedmodels
from cnn_finetune import make_model


def cell_densenet(model_name='densenet121', num_classes=1108, n_channels=6, weight=None):
    model = make_model(
        model_name=model_name,
        num_classes=31,
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

    if weight:
        model_state_dict = torch.load(weight)['model_state_dict']
        model.load_state_dict(model_state_dict)
        print(f"\n\n******************************* Loaded checkpoint {weight}")

    in_features = model._classifier.in_features
    model._classifier = nn.Linear(
        in_features=in_features, out_features=num_classes
    )

    return model


if __name__ == '__main__':
    import torch
    model = cell_densenet(model_name='densenet121')
    x = torch.randn((1, 6, 320, 320))
    y = model(x)
