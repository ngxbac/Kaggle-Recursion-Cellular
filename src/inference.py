import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *

from models import *
from augmentation import *
from dataset import RecursionCellularSite


device = torch.device('cuda')


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


def predict_ds(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = [p.detach().cpu().numpy() for p in pred]
            preds.append(pred)

    preds = np.concatenate(preds, axis=1)
    print(preds.shape)
    return preds


def predict_all():
    test_csv = '/raid/data/kaggle/recursion-cellular-image-classification/test.csv'
    # test_csv = "./csv/pseudo_all/valid_0"
    # test_csv = './csv/valid_0.csv'
    model_name = 'se_resnext50_32x4d'

    channel_str = "[1,2,3,4,5]"

    scheme = "pseudoall2_from_control"

    for fold in [
        0
    ]:

        log_dir = f"/raid/bac/kaggle/logs/recursion_cell/{scheme}/{channel_str}/fold_{fold}/{model_name}/"
        root = "/raid/data/kaggle/recursion-cellular-image-classification/"
        sites = [1]
        channels = [int(i) for i in channel_str[1:-1].split(',')]

        ckp = os.path.join(log_dir, "checkpoints/best.pth")
        model = cell_senet(
            model_name=model_name,
            num_classes=1108,
            n_channels=len(channels) * len(sites),
            # weight=f"/raid/bac/kaggle/logs/recursion_cell/test/190826/controls/{model_name}/checkpoints/best.pth",
        )

        checkpoint = f"{log_dir}/checkpoints/best.pth"
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model = nn.DataParallel(model)

        print("*" * 50)
        print(f"checkpoint: {ckp}")
        print(f"Channel: {channel_str}")
        preds = []
        for site in [1, 2]:
            # Dataset
            dataset = RecursionCellularSite(
                csv_file=test_csv,
                root=root,
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
        all_preds = np.argmax(preds, axis=1)
        df = pd.read_csv(test_csv)
        submission = df.copy()
        submission['sirna'] = all_preds.astype(int)
        os.makedirs(f"./prediction2/{scheme}/fold_{fold}/", exist_ok=True)
        submission.to_csv(f'./prediction2/{scheme}/fold_{fold}/{model_name}_{channel_str}_test.csv', index=False, columns=['id_code', 'sirna'])
        np.save(f"./prediction2/{scheme}/fold_{fold}/{model_name}_{channel_str}_test.npy", preds)


def predict_deepsupervision():
    test_csv = '/raid/data/kaggle/recursion-cellular-image-classification/test.csv'
    # test_csv = './csv/valid_0.csv'
    model_name = 'DSInceptionV3'
    experiment = '6channels_sgd'

    for channel_str in [
        "[1,2,3,4,5,6]",
    ]:

        log_dir = f"/raid/bac/kaggle/logs/recursion_cell/test/190731/fold_0/{model_name}/"
        root = "/raid/data/kaggle/recursion-cellular-image-classification/"
        sites = [1]
        channels = [int(i) for i in channel_str[1:-1].split(',')]

        # log_dir = log_dir.replace('[', '[[]')
        # log_dir = log_dir.replace(']', '[]]')

        ckp = os.path.join(log_dir, "checkpoints/best.pth")
        model = DSInceptionV3(
            num_classes=1108,
            n_channels=len(channels) * len(sites)
        )

        checkpoint = torch.load(ckp)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        # model = nn.DataParallel(model)

        print("*" * 50)
        print(f"checkpoint: {ckp}")
        print(f"Channel: {channel_str}")
        preds = []
        for site in [1, 2]:
            # Dataset
            dataset = RecursionCellularSite(
                csv_file=test_csv,
                root=root,
                transform=valid_aug(512),
                mode='test',
                sites=[site],
                channels=channels
            )

            loader = DataLoader(
                dataset=dataset,
                batch_size=128,
                shuffle=False,
                num_workers=8,
            )

            pred = predict_ds(model, loader)
            preds.append(pred)

        preds = np.asarray(preds)#.mean(axis=0)
        print(preds.shape)
        # all_preds = np.argmax(preds, axis=1)
        df = pd.read_csv(test_csv)
        submission = df.copy()
        # submission['sirna'] = all_preds.astype(int)
        os.makedirs("./prediction/DS/", exist_ok=True)
        # submission.to_csv(f'./prediction/DS/{model_name}_test.csv', index=False, columns=['id_code', 'sirna'])
        np.save(f"./prediction/DS/{model_name}_{experiment}.npy", preds)


if __name__ == '__main__':
    # predict_all()
    # predict_deepsupervision()
    predict_all()
