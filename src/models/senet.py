import torch.nn as nn
import pretrainedmodels
import torch
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
            pretrained=True
        )

        conv1 = self.model._features[0].conv1
        self.model._features[0].conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias
        )

        # copy pretrained weights
        self.model._features[0].conv1.weight.data[:,:3,:,:] = conv1.weight.data
        self.model._features[0].conv1.weight.data[:,3:n_channels,:,:] = conv1.weight.data[:,:int(n_channels-3),:,:]

        # Positive SIRNA
        self.pos_embedding = nn.Embedding(
            num_embeddings=31, embedding_dim=32
        )

        # classifiers
        in_features = self.model._classifier.in_features
        self.fc = nn.Linear(
            in_features=in_features * 2 + 32,
            out_features=num_classes
        )

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x, x_pos, x_pos_sirna):
        x = self.model._features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x_pos = self.model._features(x_pos)
        x_pos = F.adaptive_avg_pool2d(x_pos, 1)
        x_pos = x_pos.view(x_pos.size(0), -1)

        x_pos_sirna = self.pos_embedding(x_pos_sirna)
        x = torch.cat([x, x_pos, x_pos_sirna], 1)

        return self.fc(x)


class SENetWithPosFeatures(nn.Module):
    def __init__(self,  model_name="se_resnext50_32x4d",
                        num_classes=1108,
                        n_channels=6):
        super(SENetWithPosFeatures, self).__init__()

        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True
        )

        conv1 = self.model._features[0].conv1
        self.model._features[0].conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=conv1.padding,
            bias=conv1.bias
        )

        # copy pretrained weights
        self.model._features[0].conv1.weight.data[:,:3,:,:] = conv1.weight.data
        self.model._features[0].conv1.weight.data[:,3:n_channels,:,:] = conv1.weight.data[:,:int(n_channels-3),:,:]

        # Positive SIRNA
        self.pos_embedding = nn.Embedding(
            num_embeddings=31, embedding_dim=32
        )

        # classifiers
        in_features = self.model._classifier.in_features
        self.fc = nn.Linear(
            in_features=in_features * 2 + 32,
            out_features=num_classes
        )

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x, x_pos, x_pos_sirna):
        x = self.model._features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x_pos = self.model._features(x_pos)
        x_pos = F.adaptive_avg_pool2d(x_pos, 1)
        x_pos = x_pos.view(x_pos.size(0), -1)

        x_pos_sirna = self.pos_embedding(x_pos_sirna)
        x = torch.cat([x, x_pos, x_pos_sirna], 1)

        return self.fc(x)


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

    return model
