from collections import OrderedDict
import torch
import torch.nn as nn
import random
from catalyst.dl.experiment import ConfigExperiment
from dataset import *
from augmentation import train_aug, valid_aug


class Experiment(ConfigExperiment):
    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings("ignore")

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage == "stage0":
            for param in model_._features.parameters():
                param.requires_grad = False
            print("Freeze backbone model !!!")
        else:
            for param in model_._features.parameters():
                param.requires_grad = True
            print("Unfreeze backbone model !!!")

        return model_

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        """
        image_key: 'id'
        label_key: 'attribute_ids'
        """

        image_size = kwargs.get("image_size", 320)
        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        sites = kwargs.get('sites', [1])
        channels = kwargs.get('channels', [1, 2, 3, 4, 5, 6])
        root = kwargs.get('root', None)

        if train_csv:
            transform = train_aug(image_size)
            train_set = RecursionCellularSite(
                csv_file=train_csv,
                root=root,
                transform=transform,
                mode='train',
                sites=sites,
                channels=channels
            )
            datasets["train"] = train_set

        if valid_csv:
            transform = valid_aug(image_size)
            valid_set = RecursionCellularSite(
                csv_file=valid_csv,
                root=root,
                transform=transform,
                mode='train',
                sites=sites,
                channels=channels
            )
            datasets["valid"] = valid_set

        return datasets
