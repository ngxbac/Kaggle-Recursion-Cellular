from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold


classes = [
    '0', '1', '2', '3', '4', '5', '6',
    '7', '8', '9', '10', '11', '12', '13',
    '14', '15', '16', '17', '18', '19', '20',
    '21', '22', '23', '24', '25', '26', '27'
]


def main():
    train_csv = "/raid/data/kaggle/protein/train.csv"

    split_csv = "./csv/hpa"
    os.makedirs(split_csv, exist_ok=True)

    df = pd.read_csv(train_csv)
    df["is_external"] = False

    # df = pd.concat([df, hpa_df], axis=0)

    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes
    y = mlb.transform(df['Target'].str.split()).astype(np.float32)
    msss = MultilabelStratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=9669)

    for fold, (train_index, valid_index) in enumerate(msss.split(df, y)):
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]
        train_df.to_csv(os.path.join(split_csv, f"train_{fold}.csv"), index=False)
        valid_df.to_csv(os.path.join(split_csv, f"valid_{fold}.csv"), index=False)


if __name__ == '__main__':
    main()
