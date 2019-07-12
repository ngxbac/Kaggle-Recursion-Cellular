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


def predict_all():
    test_csv = '/raid/data/kaggle/recursion-cellular-image-classification/test.csv'
    # test_csv = './csv/valid_0.csv'
    model_name = 'se_resnext50_32x4d'

    tta_dict = test_tta(image_size=512)

    for channel_str in [
        "[1,2,3,4]", "[1,2,3,5]", "[1,2,3,6]",
        "[1,2,4,5]", "[1,2,4,6]", "[1,2,5,6]",
        "[1,3,4,5]", "[1,3,4,6]", "[1,3,5,6]",
        "[1,4,5,6]", "[2,3,4,5]", "[2,3,4,6]",
        "[2,3,5,6]", "[2,4,5,6]", "[3,4,5,6]"
    ]:

        log_dir = f"/raid/bac/kaggle/logs/recursion_cell/search_channels/{channel_str}/{model_name}/"
        root = "/raid/data/kaggle/recursion-cellular-image-classification/"
        sites = [1]
        channels = [int(i) for i in channel_str[1:-1].split(',')]

        log_dir = log_dir.replace('[', '[[]')

        all_checkpoints = glob.glob(f"{log_dir}/checkpoints/stage1.*")
        for ckp_num, ckp in enumerate(all_checkpoints):
            model = cell_senet(
                model_name="se_resnext50_32x4d",
                num_classes=1108,
                n_channels=len(channels) * len(sites)
            )

            checkpoint = torch.load(ckp)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            # model = nn.DataParallel(model)

            for tta_key, tta_value in tta_dict.items():
                print("*" * 50)
                print(f"checkpoint: {ckp}")
                print(f"Channel: {channel_str}")
                print(f"TTA: {tta_key}")
                preds = []
                for site in [1, 2]:
                    # Dataset
                    dataset = RecursionCellularSite(
                        csv_file=test_csv,
                        root=root,
                        transform=tta_value,
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

                    pred = predict(model, loader)
                    preds.append(pred)

                preds = np.asarray(preds).mean(axis=0)
                all_preds = np.argmax(preds, axis=1)
                df = pd.read_csv(test_csv)
                submission = df.copy()
                submission['sirna'] = all_preds.astype(int)
                os.makedirs("./prediction/fold_0/", exist_ok=True)
                submission.to_csv(f'./prediction/fold_0/{model_name}_{channel_str}_ckp{ckp_num}_tta_{tta_key}_test.csv', index=False, columns=['id_code', 'sirna'])
                np.save(f"./prediction/fold_0/{model_name}_{channel_str}_ckp{ckp_num}_tta_{tta_key}_test.npy", preds)


if __name__ == '__main__':
    predict_all()
