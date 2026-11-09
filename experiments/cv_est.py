"""Blend CV (n-gram + Qwen-emb L14 + BERT L6) where prompt removal uses ESTIMATED groups.

Usage: python3 cv_est.py SCRATCH_DIR
Groups come only from text (no prompt_id, no labels). Folds still split by TRUE prompt so nothing leaks.
Centring/assignment for a held-out fold uses only that fold's rows, as at test time.
"""
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler, normalize

from common import load, folds, bacc, center_by_group, center_in_folds, assign_in_folds

S = sys.argv[1]
d, y, true_g, classes = load()
splits = folds(y, true_g)
Eq, Eb = np.load(f"{S}/emb_qwen.npy"), np.load(f"{S}/emb_bert.npy")
ng = np.load(f"{S}/oof_ngram.npy")

q = Eq[:, 1]
topic = normalize(q - q.mean(0))


def ward(K):
    return AgglomerativeClustering(n_clusters=K, linkage="ward").fit(topic).labels_


def lr_scores(X, g):
    oof = np.zeros((len(y), len(classes)))
    for tr, te in splits:
        Xtr, Xte = center_by_group(X[tr], g[tr]), center_by_group(X[te], g[te])
        sc = StandardScaler().fit(Xtr)
        oof[te] = LogisticRegression(C=0.01, max_iter=400).fit(sc.transform(Xtr), y[tr]).decision_function(sc.transform(Xte))
    return oof


sources = {"true prompts": true_g}
for K in (150, 110, 200):
    sources[f"estimated, K={K}"] = ward(K)

print(f"{'groups':<22} ARI    | qwen  bert  ngram | blend argmax | blend + assignment")
for name, g in sources.items():
    ari = adjusted_rand_score(true_g, g)
    parts = [lr_scores(Eq[:, 0], g), lr_scores(Eb[:, 0], g), center_in_folds(ng, g, splits)]
    parts = [p / p.std() for p in parts]
    blend = sum(parts)
    print(f"{name:<22} {ari:.3f}  | {bacc(y, parts[0].argmax(1)):.3f} {bacc(y, parts[1].argmax(1)):.3f} {bacc(y, parts[2].argmax(1)):.3f} |"
          f"    {bacc(y, blend.argmax(1)):.3f}     |     {bacc(y, assign_in_folds(blend, g, splits)):.3f}", flush=True)
