"""Estimate prompt groups from text alone (no prompt_id, no labels) and score them against the truth.

Usage: python3 est_groups.py EMB_QWEN.npy OUT_DIR
Saves OUT_DIR/groups_<method>.npy for each method.
"""
import sys
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from common import load

emb, out = sys.argv[1], sys.argv[2]
d, y, groups, classes = load()
N = len(d)
K = int(round(N / 18))

feats = {}
tf = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True, min_df=2, max_df=0.3)
feats["tfidf"] = normalize(tf.fit_transform(d.text)).toarray()
q = np.load(emb)[:, 1]                       # Qwen3-Embedding last layer, mean pooled: semantic, not stylistic
feats["qwen_last"] = normalize(q - q.mean(0))
feats["both"] = normalize(np.hstack([feats["tfidf"], feats["qwen_last"]]))

print(f"{N} rows, true prompts {len(np.unique(groups))}, asking for K={K} clusters")
for name, X in feats.items():
    for link in ("average", "ward"):
        if link == "ward":
            cl = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X)
        else:
            cl = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit(X)
        lab = cl.labels_
        sizes = np.bincount(lab)
        # pairwise precision: of passages put together, how many share a true prompt
        ari = adjusted_rand_score(groups, lab)
        pure = np.mean([np.bincount(pd_ := np.unique(groups[lab == c], return_inverse=True)[1]).max() / (lab == c).sum()
                        for c in np.unique(lab)])
        print(f"  {name:<9} {link:<8} ARI {ari:.3f}  mean cluster purity {pure:.3f}  sizes min/med/max {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
        np.save(f"{out}/groups_{name}_{link}.npy", lab)
