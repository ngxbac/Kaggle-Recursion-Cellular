import pandas as pd
import numpy as np

import torch
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
            pred = Ftorch.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


def predict_all():
    test_csv = '/raid/data/kaggle/recursion-cellular-image-classification/test.csv'
    # test_csv = './csv/valid_0.csv'

    experiment = 'c123_s12_smooth_nadam_'
    model_name = 'se_resnext50_32x4d'

    log_dir = f"/raid/bac/kaggle/logs/recursion_cell/test/{experiment}/{model_name}/"
    root = "/raid/data/kaggle/recursion-cellular-image-classification/"
    sites = [1, 2]
    channels = [1,2,3]

    model = cell_senet(
        model_name="se_resnext50_32x4d",
        num_classes=1108,
        n_channels=len(channels) * len(sites)
    )

    checkpoint = f"{log_dir}/checkpoints/best.pth"
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # Dataset
    dataset = RecursionCellularSite(
        csv_file=test_csv,
        root=root,
        transform=valid_aug(512),
        mode='test',
        sites=sites,
        channels=channels
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    pred = predict(model, loader)

    all_preds = np.argmax(pred, axis=1)
    df = pd.read_csv(test_csv)
    submission = df.copy()
    submission['sirna'] = all_preds.astype(int)
    os.makedirs("submission", exist_ok=True)
    submission.to_csv(f'./submission/{model_name}_{experiment}.csv', index=False, columns=['id_code', 'sirna'])
    np.save(f"./submission/{model_name}_{experiment}.npy", pred)


if __name__ == '__main__':
    predict_all()
