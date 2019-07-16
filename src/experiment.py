from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import random
from catalyst.dl.experiment import ConfigExperiment
import dataset as cell_dataset
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
            if hasattr(model_, 'freeze'):
                model_.freeze()
                print("Freeze backbone model !!!")
            else:
                for param in model_._features.parameters():
                    param.requires_grad = False
                print("Freeze backbone model !!!")

        else:
            if hasattr(model_, 'unfreeze'):
                model_.unfreeze()
                print("Unfreeze backbone model !!!")
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

        dataset_params = kwargs.get('dataset_params', None)
        assert dataset_params is not None

        dataset_name = dataset_params.get('dataset_name', 'RecursionCellularSite')
        dataset_func = getattr(cell_dataset, dataset_name)

        image_size = dataset_params.get("image_size", 320)
        train_csv = dataset_params.get('train_csv', None)
        valid_csv = dataset_params.get('valid_csv', None)
        sites = dataset_params.get('sites', [1])
        channels = dataset_params.get('channels', [1, 2, 3, 4, 5, 6])
        root = dataset_params.get('root', None)

        if train_csv:
            transform = train_aug(image_size)
            train_set = dataset_func(
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
            valid_set = dataset_func(
                csv_file=valid_csv,
                root=root,
                transform=transform,
                mode='train',
                sites=sites,
                channels=channels
            )
            datasets["valid"] = valid_set

        return datasets
