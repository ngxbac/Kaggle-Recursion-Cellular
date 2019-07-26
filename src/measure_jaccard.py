import pandas as pd
import numpy as np
import click
from ortools.graph import pywrapgraph
from scipy.special import softmax
from tqdm import *


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def get_group(sirna_group, group_label_dict):
    scores = []
    for k, v in group_label_dict.items():
        scores.append(get_jaccard_sim(sirna_group, v))
    return np.argmax(scores)


def get_group_score(sirna_group, group_label_dict):
    scores = []
    for k, v in group_label_dict.items():
        scores.append(get_jaccard_sim(sirna_group, v))
    return np.max(scores)


def mcf_cal(X, dict_dist):
    X = X / X.sum(axis=0)
    m = X * 1000000000
    m = m.astype(np.int64)
    nb_rows, nb_classes = X.shape[0], X.shape[1]

    mcf = pywrapgraph.SimpleMinCostFlow()

    # Suppliers: distribution
    for j in range(nb_classes):
        mcf.SetNodeSupply(j + nb_rows, int(dict_dist[j]))

    # Rows
    for i in range(nb_rows):
        mcf.SetNodeSupply(i, -1)
        for j in range(nb_classes):
            mcf.AddArcWithCapacityAndUnitCost(j + nb_rows, i, 1, int(-m[i][j]))
    mcf.SolveMaxFlowWithMinCost()

    assignment = np.zeros(nb_rows, dtype=np.int32)
    for i in range(mcf.NumArcs()):
        if mcf.Flow(i) > 0:
            assignment[mcf.Head(i)] = mcf.Tail(i) - nb_rows

    return assignment


public_experiments = [
    "HEPG2-08",
    "HUVEC-17",
    "RPE-08",
    "U2OS-04",
]


@click.group()
def cli():
    print("Leader board estimation")


@cli.command()
@click.option('--csv', type=str)
@click.option('--prediction', type=str)
@click.option('--group_label', type=str)
@click.option('--is_logit', type=bool, default=True)
def measure_jaccard(
    csv,
    prediction,
    group_label,
    is_logit
):
    test_df = pd.read_csv(csv)
    pred = np.load(prediction)
    if is_logit:
        test_pred = softmax(pred, axis=1)

    # Load group label
    group_labels = np.load(group_label, allow_pickle=True)
    group_label_dict = {}
    for i, group_label in enumerate(group_labels):
        group_label_dict[i] = group_label

    expect_dist_dict = {}
    for i in range(1108):
        expect_dist_dict[i] = 1

    test_exp = test_df.experiment.unique()
    test_pred_no = np.zeros((test_df.shape[0],))
    test_pred_cal = np.zeros((test_df.shape[0],))

    for exp in tqdm(test_exp, total=len(test_exp), desc='Experiment level calibration '):
        exp_df = test_df[test_df.experiment == exp]
        exp_pred = test_pred[exp_df.index]
        exp_pred_cls = np.argmax(exp_pred, axis=1)
        test_pred_no[exp_df.index] = exp_pred_cls
        calib_cls = mcf_cal(exp_pred, expect_dist_dict)
        test_pred_cal[exp_df.index] = calib_cls

    test_df["sirna_pred"] = test_pred_no.astype(int)
    test_df["sirna_cali"] = test_pred_cal.astype(int)

    gb = test_df.groupby(['plate', 'experiment']).agg({
        'sirna_cali': ['unique']
    }).reset_index()
    gb.columns = ["plate", "experiment", 'sirna_unique']
    gb['sirna_unique'] = gb['sirna_unique'].apply(lambda x: " ".join([str(i) for i in np.sort(x)]))
    gb["group_label"] = gb["sirna_unique"].apply(lambda x: get_group(x, group_label_dict))
    gb["group_label_score"] = gb["sirna_unique"].apply(lambda x: get_group_score(x, group_label_dict))

    # Evaludate public jaccard scores
    leaderboard_jaccard(gb, leaderboard='public')
    leaderboard_jaccard(gb, leaderboard='private')


def leaderboard_jaccard(df, leaderboard='public'):
    print()
    print("**" * 50)
    if leaderboard == 'public':
        gb = df[df.experiment.isin(public_experiments)]
    else:
        gb = df[~df.experiment.isin(public_experiments)]

    avg_jaccard = gb['group_label_score'].mean()
    print("Leader board: {}, average jaccard: {}".format(leaderboard, avg_jaccard))
    gb = gb.sort_values(by="group_label_score")
    for exp, plate, score in zip(gb.experiment, gb.plate, gb.group_label_score):
        print(f"Experiment: {exp}, plate: {plate}, score: {score}")


if __name__ == '__main__':
    cli()
