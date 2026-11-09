"""Shared helpers for local prompt-grouped CV on the official train set."""
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

DATA = "/home/rohan/kagglecomp/name-that-llm/train.csv"
N_FOLDS = 5


def load():
    d = pd.read_csv(DATA)
    classes = sorted(d.model.unique())
    y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy()
    groups = d.prompt_id.to_numpy()
    return d, y, groups, classes


def folds(y, groups):
    """Whole prompts go to one fold, same split every time."""
    return list(GroupKFold(n_splits=N_FOLDS).split(np.zeros(len(y)), y, groups))


def bacc(y, pred):
    return balanced_accuracy_score(y, pred)


def center_by_group(X, groups):
    """Subtract each prompt group's mean row (works on features or scores)."""
    X = np.asarray(X, dtype=np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        X[m] -= X[m].mean(0, keepdims=True)
    return X


def assign_by_group(S, groups):
    """Hungarian assignment inside each group: each model used at most ceil(size/18) times.
    Exactly one use per model when a group has 18 rows. Only valid if the data really
    has one passage per model per prompt."""
    from scipy.optimize import linear_sum_assignment
    n_cls = S.shape[1]
    pred = S.argmax(1).copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        reps = int(np.ceil(len(idx) / n_cls))
        r, c = linear_sum_assignment(-np.tile(S[idx], (1, reps)))
        pred[idx[r]] = c % n_cls
    return pred


def center_in_folds(X, groups, splits):
    """Centre scores per group using only rows of the same held-out fold (as at test time)."""
    X = np.asarray(X, dtype=np.float64).copy()
    for _, te in splits:
        X[te] = center_by_group(X[te], groups[te])
    return X


def assign_in_folds(S, groups, splits):
    pred = np.empty(len(S), dtype=int)
    for _, te in splits:
        pred[te] = assign_by_group(S[te], groups[te])
    return pred
