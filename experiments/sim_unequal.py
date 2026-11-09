"""Does prompt removal survive a test set with UNEQUAL passages per model?  (rules: test counts per model are not equal)
Usage: python3 sim_unequal.py RESULTS_RUN_DIR SCRATCH_DIR
Models train on balanced official-train folds as before. The held-out rows are then thinned: each model keeps a random
25-100% of its rows, so prompt groups have uneven model mixes. Groups are estimated from text only (Ward on Qwen-emb last layer)
at three cluster counts, since the real number of test prompts is unknown. Assignment is NOT used (invalid with unequal counts).
Reports balanced accuracy on the thinned rows, mean +/- std over 6 random thinnings."""
import sys
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize
from common import load, folds, bacc, center_by_group, assign_by_group

run, S = Path(sys.argv[1]), sys.argv[2]
d, y, true_g, classes = load()
N = len(y); splits = folds(y, true_g)
Eq, Eb = np.load(f"{S}/emb_qwen.npy"), np.load(f"{S}/emb_bert.npy")
topic = normalize(Eq[:, 1] - Eq[:, 1].mean(0))
lora = np.mean([np.load(p)["logp"].astype(np.float64) for p in sorted(run.glob("runs/*/pred.npz"))], axis=0)
ngp = np.load(run / "ngram.npz")["logp"].astype(np.float64)
feats = {"qwen": Eq[:, 0], "bert": Eb[:, 0]}

# frozen models: one fit per fold and variant, trained on the balanced train fold (centred with TRUE groups for 'cent')
fits = {}
for name, X in feats.items():
    for var in ("raw", "cent"):
        for k, (tr, te) in enumerate(splits):
            Xtr = center_by_group(X[tr], true_g[tr]) if var == "cent" else X[tr]
            sc = StandardScaler().fit(Xtr)
            fits[name, var, k] = (sc, LogisticRegression(C=0.01, max_iter=400).fit(sc.transform(Xtr), y[tr]))
print("frozen models fitted", flush=True)

def frozen_scores(name, var, kept, est_full):
    out = np.zeros((N, 18))
    for k, (tr, te) in enumerate(splits):
        rows = np.intersect1d(te, kept)
        X = feats[name][rows]
        if var == "cent":
            X = center_by_group(X, est_full[rows])
        sc, clf = fits[name, var, k]
        out[rows] = clf.decision_function(sc.transform(X))
    return out[kept]

def evaluate(kept, K):
    est_full = np.full(N, -1)
    est_full[kept] = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(topic[kept]).labels_
    est = est_full[kept]; yk = y[kept]
    z = lambda a: a / a.std()
    res = {}
    res["LoRA raw"] = lora[kept]
    res["LoRA centred"] = center_by_group(lora[kept], est)
    raw_parts = [z(lora[kept]), z(ngp[kept])] + [z(frozen_scores(n, "raw", kept, est_full)) for n in feats]
    cen_parts = [z(center_by_group(lora[kept], est)), z(center_by_group(ngp[kept], est))] + [z(frozen_scores(n, "cent", kept, est_full)) for n in feats]
    res["blend raw"] = sum(raw_parts); res["blend centred"] = sum(cen_parts)
    fro = [z(frozen_scores(n, "cent", kept, est_full)) for n in feats]
    res["LoRA raw + ngram raw + frozen centred"] = z(lora[kept]) + z(ngp[kept]) + sum(fro)
    res["LoRA raw + ngram centred + frozen centred"] = z(lora[kept]) + z(center_by_group(ngp[kept], est)) + sum(fro)
    res["2xLoRA raw + ngram centred + frozen centred"] = 2 * z(lora[kept]) + z(center_by_group(ngp[kept], est)) + sum(fro)
    out = {k: bacc(yk, v.argmax(1)) for k, v in res.items()}
    best = res["2xLoRA raw + ngram centred + frozen centred"]
    out["BEST RECIPE + assignment inside estimated groups"] = bacc(yk, assign_by_group(best, est))
    out["LoRA raw + assignment inside estimated groups"] = bacc(yk, assign_by_group(res["LoRA raw"], est))
    return out

rows = {}
print("balanced reference (all rows kept)", flush=True)
for mult, label in ((1.0, "oracle K"), (0.7, "K x0.7"), (1.4, "K x1.4")):
    r = evaluate(np.arange(N), int(round(150 * mult)))
    print(f"  {label:<9}", "\n            ".join(f"{k}: {v:.3f}" for k, v in r.items()), flush=True)
print("unequal test, 6 thinnings")
for mult, label in ((1.0, "oracle K"), (0.7, "K x0.7"), (1.4, "K x1.4")):
    acc = {}
    for s in range(6):
        rng = np.random.default_rng(100 + s)
        f = rng.uniform(0.25, 1.0, 18)
        kept = np.where(rng.random(N) < f[y])[0]
        for k, v in evaluate(kept, int(round(150 * mult))).items():
            acc.setdefault(k, []).append(v)
    print(f"  {label:<9}", "\n            ".join(f"{k}: {np.mean(v):.3f}+/-{np.std(v):.3f}" for k, v in acc.items()), flush=True)
