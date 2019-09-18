import pandas as pd
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import json
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from tqdm import tqdm


expect_dist_dict = {}
for i in range(1108):
    expect_dist_dict[i] = 1


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


from ortools.graph import pywrapgraph
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


def _get_predicts(predicts, coefficients):
    return torch.einsum("ij,j->ij", (predicts, coefficients))


def _get_labels_distribution(predicts, coefficients):
    predicts = _get_predicts(predicts, coefficients)
    labels = predicts.argmax(dim=-1)
    counter = torch.bincount(labels, minlength=predicts.shape[1])
    return counter


def _compute_score_with_coefficients(predicts, coefficients):
    counter = _get_labels_distribution(predicts, coefficients).float()
    counter = counter * 100 / len(predicts)
    max_scores = torch.ones(len(coefficients)).cuda().float() * 100 / len(coefficients)
    result, _ = torch.min(torch.cat([counter.unsqueeze(0), max_scores.unsqueeze(0)], dim=0), dim=0)

    return float(result.sum().cpu())


def _find_best_coefficients(predicts, coefficients, alpha=0.001, iterations=100):
    best_coefficients = coefficients.clone()
    best_score = _compute_score_with_coefficients(predicts, coefficients)

    for _ in range(iterations):
        counter = _get_labels_distribution(predicts, coefficients)
        label = int(torch.argmax(counter).cpu())
        coefficients[label] -= alpha
        score = _compute_score_with_coefficients(predicts, coefficients)
        if score > best_score:
            best_score = score
            best_coefficients = coefficients.clone()

    return best_coefficients


def pavel_calib(y):
    alpha = 0.01

    coefs = torch.ones(y.shape[1]).cuda().float()
    last_score = _compute_score_with_coefficients(y, coefs)
    print("Start score", last_score)

    while alpha >= 0.0001:
        coefs = _find_best_coefficients(y, coefs, iterations=3000, alpha=alpha)
        new_score = _compute_score_with_coefficients(y, coefs)

        if new_score <= last_score:
            alpha *= 0.5

        last_score = new_score
        print("Score: {}, alpha: {}".format(last_score, alpha))

    predicts = _get_predicts(y, coefs)

    return predicts


def load_one_fold_6C5(model, fold):
    test_pred_6C5 = []
    for channel in [
        "[1,2,3,4,5]",
        "[1,2,3,4,6]",
        "[1,2,3,5,6]",
        "[1,2,4,5,6]",
        "[1,3,4,5,6]",
        "[2,3,4,5,6]",
    ]:
        pred = np.load(f"./prediction_6C5/fold_{fold}/{model}_{channel}_test.npy")
        test_pred_6C5.append(pred)
    test_pred_6C5 = np.asarray(test_pred_6C5)
    test_pred_6C5 = test_pred_6C5.mean(axis=0)
    return test_pred_6C5


def load_one_fold_6C6(model, fold):
    pred = np.load(f"../prediction_6channels/{model}_6_channel_fold{fold}_Adam.npy")
    return pred


def load_one_fold_6C6_1139(model, fold):
    pred = np.load(f"./prediction_6channels_1139/fold_{fold}/{model}_[1,2,3,4,5,6]_test.npy")
    return pred[:, :1108]


def load_kfold_6C6(model):
    pred = 0
    for fold in range(5):
        pred += load_one_fold_6C6(model, fold)
    return pred / 5


def load_kfold_6C6_1139(model):
    pred = 0
    for fold in range(5):
        pred += load_one_fold_6C6_1139(model, fold)
    return pred / 5


def load_kfold_6C5(model):
    pred = 0
    for fold in range(5):
        pred += load_one_fold_6C5(model, fold)
    return pred / 5


def load_pseudo_kfold():
    pred = 0
    for fold in range(5):
        pred += np.load(f"./prediction/pseudo/fold_{fold}/se_resnext50_32x4d_[1,2,3,4,5]_test.npy")
    pred = pred / 5
    return pred


def load_baseline_kfold():
    pred = 0
    for fold in range(5):
        pred += np.load(f"./prediction_6C5/fold_{fold}/se_resnext50_32x4d_[1,2,3,4,5]_test.npy")
    pred = pred / 5
    return pred


if __name__ == '__main__':
    model = "se_resnext50_32x4d"

    resnext101_6C4_new = np.load("test_pred_se_resnext101_32x4d_14runs_fold4.npy")
    kfold_6channels = load_kfold_6C6(model)
    kfold_6C5 = load_kfold_6C5(model)
    posneg_6C6_1139 = load_kfold_6C6_1139(model)
    posneg_1139 = np.load("../submission/se_resnext50_32x4d_1139classes_1108.npy")
    posneg_1139_2 = np.load("../submission/se_resnext50_32x4d_c1234_s1_affine_warmup_1139.npy")
    posneg_1139_2 = posneg_1139_2[:, :1108]

    yu4u_logits = 0
    yu4u_logits += np.load("../yu4u_logits/seed_0.npy")
    yu4u_logits += np.load("../yu4u_logits/seed_1.npy")
    yu4u_logits += np.load("../yu4u_logits/seed_2.npy")
    yu4u_logits /= 3

    kfold_pseudo_12345 = load_pseudo_kfold()
    kfold_baseline_12345 = load_baseline_kfold()

    baseline = np.load("./prediction_6C5/fold_0/se_resnext50_32x4d_[1,2,3,4,5]_test.npy")
    pseudo_0 = np.load("./submission/se_resnext50_32x4d_pseudo.npy")
    dontdrop = np.load("./submission/se_resnext50_32x4d_dont_drop_1.npy")

    test_df = pd.read_csv("/raid/data/kaggle/recursion-cellular-image-classification/test.csv")
    test_pred = softmax(kfold_pseudo_12345, axis=1)

    # Load group label
    group_labels = np.load("group_labels.npy", allow_pickle=True)
    group_label_dict = {}
    for i, group_label in enumerate(group_labels):
        group_label_dict[i] = group_label

    expect_dist_dict = {}
    for i in range(1108):
        expect_dist_dict[i] = 1

    test_exp = test_df.experiment.unique()
    test_pred_no = np.zeros((test_df.shape[0],))
    test_pred_cal = np.zeros((test_df.shape[0],))

    for exp in tqdm.tqdm(test_exp, total=len(test_exp), desc='Experiment level calibration '):
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

    prob = test_pred

    with open('output.json', 'r') as f:
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
        plate_prob = prob[start_id:end_id, group_labels]
        #     plate_prob = softmax(plate_prob, axis=1)
        plate_prob = plate_prob / plate_prob.sum(axis=0, keepdims=True)
        # TODO: adjust normalization degree
        #     print(plate_prob.shape)
        row_ind, col_ind = linear_sum_assignment(1 - plate_prob)
        col_ind = np.array(group_labels)[col_ind]
        sirnas.extend(col_ind)

    sub = pd.DataFrame.from_dict(
        data={"id_code": id_codes, "sirna": sirnas}
    )
    sub.to_csv("./submission/kfold_pseudo_12345.csv", index=False)
