"""Apply prompt removal to the Unsloth notebook's saved scores.

Usage: python3 apply_prompt_removal.py WORK_DIR EMB_QWEN.npy
  WORK_DIR  the notebook's work folder: runs/*/pred.npz (key logp) and ngram.npz, rows in official train.csv order
  EMB_QWEN  emb_qwen.npy from embed.py (Qwen3-Embedding, layers 14,28), used only to estimate prompt groups
Groups are estimated from text alone (Ward clustering, K = rows / 18), so this mirrors what the hidden test allows.
Assignment assumes one passage per model per prompt, which is unverified for the hidden test.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

from common import load, bacc, center_by_group, assign_by_group

work, emb = Path(sys.argv[1]), sys.argv[2]
d, y, true_g, classes = load()
q = np.load(emb)[:, 1]
est = AgglomerativeClustering(n_clusters=int(round(len(d) / 18)), linkage="ward").fit(normalize(q - q.mean(0))).labels_
print(f"estimated groups: ARI vs true prompts {adjusted_rand_score(true_g, est):.3f}")

scores = {p.parent.name: np.load(p)["logp"].astype(np.float64) for p in sorted((work / "runs").glob("*/pred.npz"))}
if (work / "ngram.npz").exists():
    scores["ngram"] = np.load(work / "ngram.npz")["logp"].astype(np.float64)
if not scores:
    sys.exit("no pred.npz or ngram.npz found")
n = min(len(v) for v in scores.values())
if n != len(y):
    print(f"WARNING: only {n} of {len(y)} rows scored (smoke run?). Using the first {n}; groups were estimated on all rows.")
y, true_g, est = y[:n], true_g[:n], est[:n]
scores = {k: v[:n] for k, v in scores.items()}
scores["blend of all"] = sum(scores.values()) / len(scores)

print(f"\n{'':<16} raw   | true groups: centred assign | estimated groups: centred assign")
for k, S in scores.items():
    ct, ce = center_by_group(S, true_g), center_by_group(S, est)
    print(f"{k:<16} {bacc(y, S.argmax(1)):.3f} |              {bacc(y, ct.argmax(1)):.3f}   {bacc(y, assign_by_group(ct, true_g)):.3f} |"
          f"                  {bacc(y, ce.argmax(1)):.3f}   {bacc(y, assign_by_group(ce, est)):.3f}")
