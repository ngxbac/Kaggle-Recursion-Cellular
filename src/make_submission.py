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
    log_dir = "/raid/bac/kaggle/logs/recursion_cell/test/c123_s1_1cycle_adamw/se_resnext50_32x4d/"
    root = "/raid/data/kaggle/recursion-cellular-image-classification/"
    site = 1
    channels = [1,2,3]

    model = cell_senet(
        model_name="se_resnext50_32x4d",
        num_classes=1108,
        n_channels=len(channels)
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
        site=site,
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
    submission.to_csv('./submission/se_resnext50_32x4d_c123_s1_1cycle_adamw.csv', index=False, columns=['id_code', 'sirna'])
    np.save("./submission/se_resnext50_32x4d_c123_s1_1cycle_adamw.npy", pred)


if __name__ == '__main__':
    predict_all()
