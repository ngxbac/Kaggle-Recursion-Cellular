import torch
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


class SENetGrouplevel(nn.Module):
    def __init__(self,  model_name="seresnext26_32x4d",
                        num_classes=1108,
                        n_channels=6):
        super(SENetGrouplevel, self).__init__()

        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout_p=0.3
        )
        print("*" * 100)
        print("SENetGrouplevel")
        conv1 = self.model._features[0].conv1
        self.model._features[0].conv1 = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model._features[0].conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model._features[0].conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

        self.group_label_embedding = nn.Embedding(num_embeddings=4, embedding_dim=8)

        in_features = self.model._classifier.in_features
        self.final_fc = nn.Linear(
            in_features=in_features + 8, out_features=num_classes
        )

    def forward(self, x, group_label):
        features = self.model._features(x)
        features = self.model.pool(features)
        features = features.view(features.size(0), -1)

        group_embedding = self.group_label_embedding(group_label)
        features = torch.cat([
            features, group_embedding
        ], 1)

        return self.final_fc(features)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class SENetCellType(nn.Module):
    def __init__(self,  model_name="seresnext26_32x4d",
                        num_classes=1108,
                        n_channels=6):
        super(SENetCellType, self).__init__()

        self.model = make_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=True,
            dropout_p=0.3
        )
        print("*" * 100)
        print("SENetGrouplevel")
        conv1 = self.model._features[0].conv1
        self.model._features[0].conv1 = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model._features[0].conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model._features[0].conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

        in_features = self.model._classifier.in_features
        self.final_sirna = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

        self.final_cell_type = nn.Linear(
            in_features=in_features, out_features=4
        )

    def forward(self, x):
        features = self.model._features(x)
        features = self.model.pool(features)
        features = features.view(features.size(0), -1)

        return self.final_sirna(features), self.final_cell_type(features)

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


class SENetCellMultipleDropout(nn.Module):
    def __init__(self,  model_name="seresnext26_32x4d",
                        num_classes=1108,
                        n_channels=6,
                        num_samples=4,
                        weight=None):
        super(SENetCellMultipleDropout, self).__init__()

        self.model = make_model(
            model_name=model_name,
            num_classes=31,
            pretrained=True
        )
        print("*" * 100)
        print("SENetGrouplevel")
        conv1 = self.model._features[0].conv1
        self.model._features[0].conv1 = nn.Conv2d(in_channels=n_channels,
                                             out_channels=conv1.out_channels,
                                             kernel_size=conv1.kernel_size,
                                             stride=conv1.stride,
                                             padding=conv1.padding,
                                             bias=conv1.bias)

        # copy pretrained weights
        self.model._features[0].conv1.weight.data[:, :3, :, :] = conv1.weight.data
        self.model._features[0].conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

        if weight:
            model_state_dict = torch.load(weight)['model_state_dict']
            self.model.load_state_dict(model_state_dict)
            print(f"\n\n******************************* Loaded checkpoint {weight}")

        in_features = self.model._classifier.in_features
        self.num_samples = num_samples

        self.classifier = nn.Linear(
            in_features, num_classes
        )

    def forward(self, x):
        features = self.model._features(x)
        features_flip = torch.flip(features, dims=[3])

        features_flip = self.model.pool(features_flip)
        features_flip = features_flip.view(features_flip.size(0), -1)

        features = self.model.pool(features)
        features = features.view(features.size(0), -1)

        out_logits = []
        for i in range(self.num_samples):
            if i % 2 == 0:
                feature = F.dropout(features, p=0.3)
            else:
                feature = F.dropout(features_flip, p=0.3)
            logits = self.classifier(feature)
            out_logits.append(logits)
        return out_logits

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True


def cell_senet(model_name='se_resnext50_32x4d', num_classes=1108, n_channels=6, weight=None):
    model = make_model(
        model_name=model_name,
        num_classes=31,
        pretrained=True,
        # pool=GlobalConcatPool2d(),
        # classifier_factory=make_classifier
        dropout_p=0.3
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

    if weight:
        model_state_dict = torch.load(weight)['model_state_dict']
        model.load_state_dict(model_state_dict)
        print(f"\n\n******************************* Loaded checkpoint {weight}")
        in_features = model._classifier.in_features
        model._classifier = nn.Linear(
            in_features=in_features, out_features=num_classes
        )

    return model
