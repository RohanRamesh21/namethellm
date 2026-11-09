"""How much does the native expert's centred head help when prompt groups are ESTIMATED from text (as at test time) instead of known?
Uses the author's stored out-of-fold logits for the 2,699 official rows (public_assets/native_q8/oof_reference.npz) and the cached Qwen3-Embedding
features. Each fold's 540 held-out rows are clustered on their own, like the test set. The centred head's bias is taken as 0 (not downloaded)."""
import json, sys
import numpy as np, pandas as pd
from scipy.special import log_softmax
sys.path.insert(0, 'experiments'); sys.path.insert(0, '/tmp')
import importlib.util
spec = importlib.util.spec_from_file_location('P', 'experiments/ntl_pipeline.py'); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)

A = 'public_assets/native_q8'
d = pd.read_csv('name-that-llm/train.csv'); classes = sorted(d.model.unique()); y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy(); g_true = d.prompt_id.to_numpy()
ref = np.load(f'{A}/oof_reference.npz'); plain, cent = ref['plain'], ref['centered']
q = np.load('experiments/cache/emb_qwen.npy')[:, 1]
rng = np.random.default_rng(0)

def rec(yy, p): return float(np.mean([(p[yy == c] == c).mean() for c in np.unique(yy)]))
def fuse(pl, ce, groups, cal, bias=0.0, centred_weight=None):
    n = np.unique(groups, return_inverse=True)[1]; cnt = np.bincount(n); ok = (cnt[n] >= 2) & (cnt[n] <= 18)
    mean = np.zeros((len(cnt), 18)); np.add.at(mean, n, ce); mean /= cnt[:, None]
    a = log_softmax(pl * cal['plain_scale'], axis=1); b = log_softmax((ce - mean[n] + bias) * cal['centered_scale'], axis=1)
    w = cal['selected_centered_weight'] if centred_weight is None else centred_weight
    return log_softmax((1 - np.where(ok, w, 0)[:, None]) * a + np.where(ok, w, 0)[:, None] * b, axis=1)

rows_out = []
for thin in (1.0, 0.6):
    res = {k: [] for k in ('plain', 'true_groups', 'est_groups', 'no_groups_w.5')}; ys = []
    for f in range(5):
        v = np.array(json.load(open(f'{A}/fold{f}/split.json'))['validation_rows']); cal = json.load(open(f'{A}/calibration/fold{f}.json'))['models']['qwen8']
        if thin < 1:   # unequal per-model counts: keep each row with a model-specific probability between thin and 1
            keep_p = rng.uniform(thin, 1.0, 18); v = v[rng.random(len(v)) < keep_p[y[v]]]
        est, info = P.estimate_groups(q[v]); yy = y[v]
        res['plain'].append((yy, log_softmax(plain[v] * cal['plain_scale'], axis=1)))
        res['true_groups'].append((yy, fuse(plain[v], cent[v], g_true[v], cal)))
        res['est_groups'].append((yy, fuse(plain[v], cent[v], est, cal)))
        res['no_groups_w.5'].append((yy, fuse(plain[v], cent[v], np.zeros(len(v), int), cal, centred_weight=0.5)))
        print(f'thin {thin} fold {f}: rows {len(v)}, est K {info["K"]} (true {len(np.unique(g_true[v]))}), sizes {info.get("size_min")}-{info.get("size_max")}', flush=True)
    for k, lst in res.items():
        yy = np.concatenate([a for a, _ in lst]); S = np.vstack([b for _, b in lst])
        per = [rec(a, b.argmax(1)) for a, b in lst]
        rows_out.append((thin, k, rec(yy, S.argmax(1)), np.std(per)))
print(pd.DataFrame(rows_out, columns=['thin', 'variant', 'balanced_acc', 'sd_over_folds']).round(4).to_string(index=False))
