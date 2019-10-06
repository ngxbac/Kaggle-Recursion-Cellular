import pandas as pd
import numpy as np
import click
import json
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax


@click.group()
def cli():
    print("Ensemble")


def load_one_fold(predict_root, model_name, fold):
    test_preds = []
    for channel in [
        "[1,2,3,4,5]",
        "[1,2,3,4,6]",
        "[1,2,3,5,6]",
        "[1,2,4,5,6]",
        "[1,3,4,5,6]",
        "[2,3,4,5,6]",
    ]:
        pred = np.load(f"{predict_root}/{channel}/fold_{fold}/{model_name}/pred_test.npy")
        test_preds.append(pred)
    test_preds = np.asarray(test_preds)
    test_preds = test_preds.mean(axis=0)
    return test_preds


def load_kfold(predict_root, model_name):
    preds = 0
    for fold in range(5):
        preds += load_one_fold(predict_root, model_name, fold) / 5
    return preds


@cli.command()
@click.option('--data_root', type=str, default='/data/')
@click.option('--predict_root', type=str, default='/logs/pseudo/')
@click.option('--group_json', type=str, default='group.json')
def ensemble(
    data_root='/data/',
    predict_root='/logs/pseudo/',
    group_json="group.json",
):
    model_names = ['se_resnext50_32x4d']
    ensemble_preds = 0
    for model_name in model_names:
        ensemble_preds += load_kfold(predict_root, model_name)

    # Just a maigc
    ensemble_preds = ensemble_preds / 121

    test_df = pd.read_csv(f"{data_root}/test.csv")
    ensemble_preds = softmax(ensemble_preds, axis=1)

    with open(group_json, 'r') as f:
        m = json.load(f)

    id_codes = test_df.id_code.values
    test_plate_id_to_group_id = m["test_plate_id_to_group_id"]
    label_group_list = m["label_group_list"]

    plate_ids = [id_code[:-4] for id_code in id_codes]
    start_indices = sorted([plate_ids.index(experiment_id) for experiment_id in set(plate_ids)])
    start_indices.append(len(plate_ids))
    sirnas = []

    for i in range(len(start_indices) - 1):
        start_id = start_indices[i]
        end_id = start_indices[i + 1]
        test_plate_id = id_codes[start_id][:-4]
        label_group_id = test_plate_id_to_group_id[test_plate_id]
        group_labels = label_group_list[label_group_id]
        plate_prob = ensemble_preds[start_id:end_id, group_labels]
        plate_prob = plate_prob / plate_prob.sum(axis=0, keepdims=True)
        row_ind, col_ind = linear_sum_assignment(1 - plate_prob)
        col_ind = np.array(group_labels)[col_ind]
        sirnas.extend(col_ind)

    sub = pd.DataFrame.from_dict(
        data={"id_code": id_codes, "sirna": sirnas}
    )
    sub.to_csv(f"{predict_root}/submission.csv", index=False)


if __name__ == '__main__':
    cli()
