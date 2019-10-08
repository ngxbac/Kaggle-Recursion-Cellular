import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import click
from tqdm import *

from models import *
from augmentation import *
from dataset import RecursionCellularSite


device = torch.device('cuda')


@click.group()
def cli():
    print("Inference")


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            # pred = Ftorch.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


@cli.command()
@click.option('--data_root', type=str, default='/data/')
@click.option('--model_root', type=str, default='/logs/pseudo/')
@click.option('--model_name', type=str, default='se_resnext50_32x4d')
@click.option('--out_dir', type=str, default='./predictions/pseudo/')
def predict_all(
    data_root='/data/',
    model_root='/logs/pseudo/',
    model_name='se_resnext50_32x4d',
    out_dir='./predictions/pseudo/'
):
    test_csv = f'{data_root}/test.csv'

    assert model_name in ["se_resnext50_32x4d", "se_resnext101_32x4d", "densenet121"]

    for channel_str in [
        "[1,2,3,4,5]",
        "[1,2,3,4,6]",
        "[1,2,3,5,6]",
        "[1,2,4,5,6]",
        "[1,3,4,5,6]",
        "[2,3,4,5,6]",
    ]:
        for fold in [0, 1, 2, 3, 4]:
            log_dir = f"{model_root}/{channel_str}/fold_{fold}/{model_name}/"
            sites = [1]
            channels = [int(i) for i in channel_str[1:-1].split(',')]

            ckp = os.path.join(log_dir, "checkpoints/best.pth")

            if model_name in ["se_resnext50_32x4d", "se_resnext101_32x4d"]:
                model = cell_senet(
                    model_name=model_name,
                    num_classes=1108,
                    n_channels=len(channels) * len(sites),
                )
            else:
                model = cell_densenet(
                    model_name=model_name,
                    num_classes=1108,
                    n_channels=len(channels) * len(sites),
                )

            checkpoint = f"{log_dir}/checkpoints/best.pth"
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model = nn.DataParallel(model)

            print("*" * 50)
            print(f"checkpoint: {ckp}")
            # print(f"Channel: {channel_str}")
            preds = []
            for site in [1, 2]:
                # Dataset
                dataset = RecursionCellularSite(
                    csv_file=test_csv,
                    root=data_root,
                    transform=valid_aug(512),
                    mode='test',
                    sites=[site],
                    channels=channels,
                    site_mode="one",
                )

                loader = DataLoader(
                    dataset=dataset,
                    batch_size=128,
                    shuffle=False,
                    num_workers=4,
                )

                pred = predict(model, loader)
                preds.append(pred)

            preds = np.asarray(preds).mean(axis=0)
            os.makedirs(f"{out_dir}/{channel_str}/fold_{fold}/{model_name}/", exist_ok=True)
            np.save(f"{out_dir}/{channel_str}/fold_{fold}/{model_name}/pred_test.npy", preds)


if __name__ == '__main__':
    cli()
