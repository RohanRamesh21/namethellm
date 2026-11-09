"""n-gram SVC under prompt-grouped CV, plus prompt removal applied to its scores.

Usage: python3 ngram_cv.py OUT_DIR
Writes OUT_DIR/oof_ngram.npy (out-of-fold decision scores, rows in train.csv order).
"""
import sys
import time
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from common import load, folds, bacc, center_by_group, assign_by_group

out = sys.argv[1]
d, y, groups, classes = load()
texts = d.text.to_numpy()
oof = np.zeros((len(y), len(classes)))

t0 = time.time()
for k, (tr, te) in enumerate(folds(y, groups)):
    word = TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char = TfidfVectorizer(lowercase=False, analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                           sublinear_tf=True, max_features=300_000)
    Xtr = sp.hstack([word.fit_transform(texts[tr]), char.fit_transform(texts[tr])]).tocsr()
    Xte = sp.hstack([word.transform(texts[te]), char.transform(texts[te])]).tocsr()
    clf = LinearSVC(C=0.5).fit(Xtr, y[tr])
    oof[te] = clf.decision_function(Xte)
    print(f"fold {k} acc {bacc(y[te], oof[te].argmax(1)):.3f}  ({time.time()-t0:.0f}s)", flush=True)

np.save(f"{out}/oof_ngram.npy", oof)
print("\nn-gram SVC, prompt-grouped 5-fold balanced accuracy")
print(f"  raw argmax                       {bacc(y, oof.argmax(1)):.3f}")
print(f"  scores centred per prompt        {bacc(y, center_by_group(oof, groups).argmax(1)):.3f}")
print(f"  one-model-per-prompt assignment  {bacc(y, assign_by_group(oof, groups)):.3f}")
