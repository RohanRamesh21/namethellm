"""Tune the weight between our Gemma stage-2 scores and the native expert on the 539 rows that stage 2 held out (30 prompts).
The expert's scores for these rows are its stored out-of-fold logits (each row comes from the fold adapter that never saw its prompt)."""
import json, sys
import numpy as np, pandas as pd
from scipy.special import log_softmax
import importlib.util
spec = importlib.util.spec_from_file_location('P', 'experiments/ntl_pipeline.py'); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
A = 'public_assets/native_q8'
d = pd.read_csv('name-that-llm/train.csv'); classes = sorted(d.model.unique()); y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy(); g = d.prompt_id.to_numpy()
prompts = np.array(sorted(d.prompt_id.unique())); rng = np.random.default_rng(0); val_prompts = set(rng.choice(prompts, 30, replace=False)); is_val = d.prompt_id.isin(val_prompts).to_numpy()
idx = np.where(is_val)[0]; G = np.load('kaggle_results_stage2_pred/stage2/runs/fold/pred.npz')['logp'].astype(np.float64); assert len(G) == len(idx), (len(G), len(idx))
yv = y[idx]
def rec(yy, p): return float(np.mean([(p[yy == c] == c).mean() for c in np.unique(yy)]))
print('Gemma stage 2 on its held-out rows:', round(rec(yv, G.argmax(1)), 4), '(report says 0.5673)')
ref = np.load(f'{A}/oof_reference.npz'); fold_of = np.zeros(len(d), int)
for f in range(5): fold_of[json.load(open(f'{A}/fold{f}/split.json'))['validation_rows']] = f
cals = [json.load(open(f'{A}/calibration/fold{f}.json'))['models']['qwen8'] for f in range(5)]
ps = np.array([cals[f]['plain_scale'] for f in fold_of[idx]])[:, None]; cs = np.array([cals[f]['centered_scale'] for f in fold_of[idx]])[:, None]; cw = np.array([cals[f]['selected_centered_weight'] for f in fold_of[idx]])[:, None]
pl, ce = ref['plain'][idx], ref['centered'][idx]
q = np.load('experiments/cache/emb_qwen.npy')[:, 1][idx]; est, info = P.estimate_groups(q); print('estimated groups on these rows:', info['K'], 'clusters (true 30)')
def native(groups, use_centred=True):
    n = np.unique(groups, return_inverse=True)[1]; cnt = np.bincount(n); ok = ((cnt[n] >= 2) & (cnt[n] <= 18))[:, None]
    mean = np.zeros((len(cnt), 18)); np.add.at(mean, n, ce); mean /= cnt[:, None]
    a = log_softmax(pl * ps, axis=1); b = log_softmax((ce - mean[n]) * cs, axis=1); w = np.where(ok, cw, 0) if use_centred else 0
    return log_softmax((1 - w) * a + w * b, axis=1)
variants = {'native plain only': native(est, False), 'native + centred (est groups)': native(est), 'native + centred (true groups)': native(g[idx])}
for k, v in variants.items(): print(f'{k:34s} {rec(yv, v.argmax(1)):.4f}')
z = lambda a: a / a.std()
print('\nweight on native in z(G)*(1-w) + z(N)*w, N = native + centred (est groups):')
keys = {p: np.where(g[idx] == p)[0] for p in np.unique(g[idx])}; brng = np.random.default_rng(1)
boot = [np.concatenate([keys[k] for k in brng.choice(list(keys), len(keys))]) for _ in range(300)]
for name, N in (('est groups', variants['native + centred (est groups)']), ('plain only', variants['native plain only'])):
    print(name)
    base = [rec(yv[b], G[b].argmax(1)) for b in boot]
    for w in (0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        S = (1 - w) * z(G) + w * z(N); s = rec(yv, S.argmax(1)); delta = [rec(yv[b], S[b].argmax(1)) - x for b, x in zip(boot, base)]
        print(f'  w={w:.1f}  {s:.4f}   vs Gemma alone {np.mean(delta):+.3f} ({np.percentile(delta, 2.5):+.3f} to {np.percentile(delta, 97.5):+.3f})')
