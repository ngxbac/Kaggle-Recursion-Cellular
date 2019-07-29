import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from tqdm import tqdm
from ortools.graph import pywrapgraph
import optuna


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


def leaderboard_jaccard(df, leaderboard='public'):
    if leaderboard == 'public':
        gb = df[df.experiment.isin(public_experiments)]
    else:
        gb = df[~df.experiment.isin(public_experiments)]

    avg_jaccard = gb['group_label_score'].mean()
    return avg_jaccard


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
    pred = np.load(f"./prediction_6channels/{model}_6_channel_fold{fold}_Adam.npy")
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


model = "se_resnext50_32x4d"
resnext101_6C4_new = np.load("./notebooks/test_pred_se_resnext101_32x4d_14runs_fold4.npy")
resnext101_6C4_old = np.load("./notebooks/test_pred_se_resnext101_32x4d_14runs.npy")

kfold_6channels = load_kfold_6C6(model)
kfold_6C5 = load_kfold_6C5(model)
posneg_6C6_1139 = load_kfold_6C6_1139(model)

posneg_1139 = np.load("./submission/se_resnext50_32x4d_1139classes_1108.npy")
posneg_1139_2 = np.load("./submission/se_resnext50_32x4d_c1234_s1_affine_warmup_1139.npy")
posneg_1139_2 = posneg_1139_2[:, :1108]

predict_array = np.asarray([
    kfold_6C5, resnext101_6C4_new, kfold_6channels,
    posneg_1139, posneg_1139_2
])

# Load group label
group_labels = np.load("./csv/group_labels.npy", allow_pickle=True)
group_label_dict = {}
for i, group_label in enumerate(group_labels):
    group_label_dict[i] = group_label

test_df = pd.read_csv("/raid/data/kaggle/recursion-cellular-image-classification/test.csv")


def measure_jaccard(predict_array, coeffects, test_df):
    test_df = test_df[test_df.experiment.isin(public_experiments)]

    avg_pred = 0
    for pred, coeff in zip(predict_array, coeffects):
        avg_pred += pred[test_df.index, :] * coeff

    test_pred = softmax(avg_pred, axis=1)

    expect_dist_dict = {}
    for i in range(1108):
        expect_dist_dict[i] = 1

    test_exp = test_df.experiment.unique()
    test_pred_no = np.zeros((test_df.shape[0],))
    test_pred_cal = np.zeros((test_df.shape[0],))
    test_df = test_df.reset_index(drop=True)

    for exp in test_exp:
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
    return leaderboard_jaccard(gb, leaderboard='public')

# Define an objective function to be minimized.
def objective(trial):

    n_models = predict_array.shape[0]
    coeffects = [trial.suggest_uniform(f'c{i}', 0.0, 1.0) for i in range(n_models)]

    jaccard = measure_jaccard(predict_array, coeffects, test_df)
    return 1-jaccard


if __name__ == '__main__':
    study = optuna.create_study()  # Create a new study.
    study.optimize(objective, n_trials=10000)  # Invoke optimization of the objective function.
