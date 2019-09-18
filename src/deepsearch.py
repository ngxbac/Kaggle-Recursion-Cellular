import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from tqdm import tqdm


expect_dist_dict = {}
for i in range(1108):
    expect_dist_dict[i] = 1


public_experiments = [
    "HEPG2-08",
    "HUVEC-17",
    "RPE-08",
    "U2OS-04",
]

pseudo_experiments = [
    "HUVEC-18", "HUVEC-17", "HEPG2-08"
]


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


def experiment_level_cal_all(test_df, pred, group_label_dict):
    test_pred = softmax(pred, axis=1)

    test_exp = test_df.experiment.unique()
    test_pred_no = np.zeros((test_df.shape[0],))
    test_pred_cal = np.zeros((test_df.shape[0],))

    for exp in tqdm(test_exp, total=len(test_exp), desc='Experiment level calibration '):
        exp_df = test_df[test_df.experiment == exp]
        exp_pred = test_pred[exp_df.index]
        exp_pred_cls = np.argmax(exp_pred, axis=1)
        test_pred_no[exp_df.index] = exp_pred_cls

        calib_pred = experiment_level_cal(exp_pred)
        score = experiment_level_score(exp_df, calib_pred, group_label_dict)
        print("Experiment: {}, scores: {}".format(exp, score))

        test_pred_cal[exp_df.index] = calib_pred

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


def experiment_level_cal(prob):
    expect_dist_dict = {}
    for i in range(1108):
        expect_dist_dict[i] = 1
    return mcf_cal(prob, expect_dist_dict)


def experiment_level_score(exp_df, exp_pred, group_label_dict):
    exp_df["sirna"] = exp_pred.astype(int)

    gb = exp_df.groupby(['plate', 'experiment']).agg({
        'sirna': ['unique']
    }).reset_index()
    gb.columns = ["plate", "experiment", 'sirna_unique']
    gb['sirna_unique'] = gb['sirna_unique'].apply(lambda x: " ".join([str(i) for i in np.sort(x)]))
    gb["group_label"] = gb["sirna_unique"].apply(lambda x: get_group(x, group_label_dict))
    gb["group_label_score"] = gb["sirna_unique"].apply(lambda x: get_group_score(x, group_label_dict))
    return gb['group_label_score'].mean()


def load_yu4u_logits():
    yu4u_logits = 0
    yu4u_logits += np.load("./yu4u_logits/seed_0.npy")
    yu4u_logits += np.load("./yu4u_logits/seed_1.npy")
    yu4u_logits += np.load("./yu4u_logits/seed_2.npy")
    yu4u_logits /= 3
    return yu4u_logits


def load_posneg_logits():
    posneg_1139 = np.load("./submission/se_resnext50_32x4d_1139classes_1108.npy")
    posneg_1139_2 = np.load("./submission/se_resnext50_32x4d_c1234_s1_affine_warmup_1139.npy")
    posneg_1139_2 = posneg_1139_2[:, :1108]
    return posneg_1139 + posneg_1139_2


class optuna_objective(object):
    def __init__(self, base_logits, model_names, preds, group_label_dict, experiment_df):
        self.base_logits = base_logits
        self.model_names = model_names
        self.preds = preds
        self.group_label_dict = group_label_dict
        self.experiment_df = experiment_df

    def __call__(self, trial):
        model_coeffs = [
            trial.suggest_categorical(f'{model_name}_coff', [0, 1])
            for model_name in self.model_names
        ]

        if self.base_logits:
            ensemble_logits = self.base_logits.copy()
        else:
            ensemble_logits = 0

        for coff, pred in zip(model_coeffs, self.preds):
            ensemble_logits += coff * pred

        ensemble_logits = ensemble_logits / 131

        ensemble_pred = softmax(ensemble_logits, axis=1)
        calib_pred = experiment_level_cal(ensemble_pred)
        score = experiment_level_score(self.experiment_df, calib_pred, self.group_label_dict)
        return score


import os
import json
import optuna


def optimize_an_experiment(base_logits, model_names, preds, test_df, group_label_dict, experiment, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    experiment_df = test_df[test_df.experiment == experiment]
    experiment_index = experiment_df.index.values
    preds = [pred[experiment_index] for pred in preds]
    if base_logits:
        base_logits = base_logits[experiment_index]

    objective = optuna_objective(base_logits, model_names, preds, group_label_dict, experiment_df)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=200)
    print(study.best_trial)

    with open(f'{out_dir}/{experiment}.json', "w") as f:
        json.dump(study.best_trial.params, f)


def ensemble_experiment_level(
    base_logits,
    model_names,
    preds,
    test_df,
    group_label_dict,
    experiment,
    out_dir,
):
    with open(f'{out_dir}/{experiment}.json', "r") as f:
        params = json.load(f)

    experiment_df = test_df[test_df.experiment == experiment]
    experiment_index = experiment_df.index.values
    preds = [pred[experiment_index] for pred in preds]
    base_logits = base_logits[experiment_index]

    model_coeffs = [
        f'{model_name}_coff'
        for model_name in model_names
    ]

    ensemble_logits = base_logits

    for coff, pred in zip(model_coeffs, preds):
        coff = params[coff]
        ensemble_logits += coff * pred

    ensemble_logits = ensemble_logits / 131
    ensemble_pred = softmax(ensemble_logits, axis=1)
    # calib_pred = experiment_level_cal(ensemble_pred)
    # score = experiment_level_score(experiment_df, calib_pred, group_label_dict)

    return ensemble_pred



if __name__ == '__main__':
    global_model_names = []
    global_model_preds = []

    yu4u_logits = load_yu4u_logits()
    global_model_names.append('yu4u')
    global_model_preds.append(yu4u_logits)

    posneg_logits = load_posneg_logits()
    global_model_names.append('posneg')
    global_model_preds.append(posneg_logits)

    base_logits = yu4u_logits + posneg_logits

    channels_6C5 = [
        "[1,2,3,4,5]",
        "[1,2,3,4,6]",
        "[1,2,3,5,6]",
        "[1,2,4,5,6]",
        "[1,3,4,5,6]",
        "[2,3,4,5,6]",
    ]

    # normal
    kfold_seresnext50_6C5 = []
    for channel in channels_6C5:
        channel_level_name = f'seresnext50_{channel}'
        preds = 0
        for fold in range(5):
            pred = np.load(f"./prediction_6C5/fold_{fold}/se_resnext50_32x4d_{channel}_test.npy")
            preds += pred
        preds = preds / 5
        kfold_seresnext50_6C5.append(preds)
        global_model_names.append(channel_level_name)
        global_model_preds.append(preds)
    kfold_seresnext50_6C5 = np.asarray(kfold_seresnext50_6C5).mean(axis=0)

    # pseudo
    kfold_seresnext50_6C5_pseudo = []
    for channel in channels_6C5:
        channel_level_name = f'seresnext50_pseudo_{channel}'
        preds = 0
        for fold in range(5):
            pred = np.load(f"./prediction/pseudo_from_control/fold_{fold}/se_resnext50_32x4d_{channel}_test.npy")
            preds += pred
        preds = preds / 5
        kfold_seresnext50_6C5_pseudo.append(preds)
        global_model_names.append(channel_level_name)
        global_model_preds.append(preds)
    kfold_seresnext50_6C5_pseudo = np.asarray(kfold_seresnext50_6C5_pseudo).mean(axis=0)

    # dense
    kfold_densenet121_6C5_pseudo = []
    for channel in [
        "[1,2,3,4,5]",
    ]:
        channel_level_name = f'densenet121_pseudo_{channel}'
        preds = 0
        for fold in range(5):
            pred = np.load(f"./prediction/pseudo_from_control/fold_{fold}/densenet121_{channel}_test.npy")
            preds += pred
        preds = preds / 5
        kfold_densenet121_6C5_pseudo.append(preds)
        global_model_names.append(channel_level_name)
        global_model_preds.append(preds)
    kfold_densenet121_6C5_pseudo = np.asarray(kfold_densenet121_6C5_pseudo).mean(axis=0)

    # Load test csv
    test_df = pd.read_csv("/raid/data/kaggle/recursion-cellular-image-classification/test.csv")

    # Load group label
    group_labels = np.load("./notebooks/group_labels.npy", allow_pickle=True)
    group_label_dict = {}
    for i, group_label in enumerate(group_labels):
        group_label_dict[i] = group_label

    kfold_model_names = [
        # 'seresnext50_6C5',
        'seresnext50_6C5_pseudo',
        'densenet121_6C5_pseudo'
    ]

    kfold_model_preds = [
        # kfold_seresnext50_6C5,
        kfold_seresnext50_6C5_pseudo,
        kfold_densenet121_6C5_pseudo
    ]

    for experiment in ['U2OS-04', 'RPE-08']:

        optimize_an_experiment(
            None,
            global_model_names,
            global_model_preds,
            test_df,
            group_label_dict,
            experiment,
            'optuna_search2'
        )

    # pred = ensemble_experiment_level(
    #     base_logits,
    #     global_model_names,
    #     global_model_preds,
    #     test_df,
    #     group_label_dict,
    #     experiment='U2OS-04',
    #     out_dir='optuna_search',
    # )
    # np.save("./notebooks/u2os-04.npy", pred)

