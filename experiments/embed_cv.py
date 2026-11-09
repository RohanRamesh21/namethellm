"""Logistic regression on frozen embeddings under prompt-grouped CV, with and without prompt removal.

Usage: python3 embed_cv.py EMB.npy TAG OUT_DIR
Variants per layer and C:
  raw        features as-is
  feat-cent  features minus their prompt-group mean (train and test each centred within their own prompts)
  score-cent raw model's scores minus per-prompt mean scores
  assign     feat-cent scores, one model per prompt (only valid if the hidden test has that structure)
Saves OOF scores of the feat-cent model to OUT_DIR/oof_TAG_L{layer}.npy for blending.
"""
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import load, folds, bacc, center_by_group, assign_by_group

emb_path, tag, out = sys.argv[1], sys.argv[2], sys.argv[3]
E = np.load(emb_path)
d, y, groups, classes = load()
splits = folds(y, groups)


def run(X, C, centre):
    oof = np.zeros((len(y), len(classes)))
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if centre:
            Xtr, Xte = center_by_group(Xtr, groups[tr]), center_by_group(Xte, groups[te])
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, max_iter=400).fit(sc.transform(Xtr), y[tr])
        oof[te] = clf.decision_function(sc.transform(Xte))
    return oof


print(f"{tag}: balanced accuracy, 5 prompt-grouped folds")
for li in range(E.shape[1]):
    X = E[:, li]
    for C in (0.01, 0.1):
        raw = run(X, C, False)
        cen = run(X, C, True)
        print(f"  layer#{li} C={C:<5} raw {bacc(y, raw.argmax(1)):.3f} | score-cent {bacc(y, center_by_group(raw, groups).argmax(1)):.3f}"
              f" | feat-cent {bacc(y, cen.argmax(1)):.3f} | assign {bacc(y, assign_by_group(cen, groups)):.3f}", flush=True)
        np.save(f"{out}/oof_{tag}_L{li}_C{C}_cent.npy", cen)
        np.save(f"{out}/oof_{tag}_L{li}_C{C}_raw.npy", raw)
