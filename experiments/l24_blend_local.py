"""Does the public Qwen3-8B-Base 'L24' expert (q8v5) add to the native Qwen3-Embedding-8B expert, out of fold, with ESTIMATED prompt groups?
Inputs: experiments/cache/q8v5_oof.npz (from kaggle_results_l24 + loss-based fold assignment), the native expert's stored OOF logits (public_assets/native_q8)."""
import json, sys
import numpy as np, pandas as pd
from scipy.special import log_softmax
import importlib.util
spec = importlib.util.spec_from_file_location('P', 'experiments/ntl_pipeline.py'); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
d = pd.read_csv('name-that-llm/train.csv'); classes = sorted(d.model.unique()); y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy(); g = d.prompt_id.to_numpy()
Q = 'public_assets/native_q8'; zq = np.load(f'{Q}/oof_reference.npz'); fq = np.zeros(len(d), int)
for f in range(5): fq[json.load(open(f'{Q}/fold{f}/split.json'))['validation_rows']] = f
cal = [json.load(open(f'{Q}/calibration/fold{f}.json'))['models']['qwen8'] for f in range(5)]
raw = np.load('experiments/cache/q8v5_oof.npz'); z8 = np.load('kaggle_results_l24/q8v5_official.npz'); fold8 = raw['fold']
# raw (uncentred) centred-head logits per row from its own held-out adapter, and that adapter's bias
C8 = z8['centred_raw'][fold8, np.arange(len(d))].astype(np.float64); B8 = z8['centred_bias'][fold8]; P8 = raw['plain'].astype(np.float64)
q = np.load('experiments/cache/emb_qwen.npy')[:, 1]
rec = lambda yy, p: float(np.mean([(p[yy == c] == c).mean() for c in np.unique(yy)]))
# groups: cluster each native fold's rows separately (540 rows, like a test chunk)
est = np.zeros(len(d), int); off = 0
for f in range(5):
    v = np.where(fq == f)[0]; lab, info = P.estimate_groups(q[v]); est[v] = lab + off; off += lab.max() + 1
def centre_lp(logits_raw, bias, groups):
    n = np.unique(groups, return_inverse=True)[1]; cnt = np.bincount(n); ok = ((cnt[n] >= 2) & (cnt[n] <= 18))[:, None]
    mean = np.zeros((len(cnt), 18)); np.add.at(mean, n, logits_raw); mean /= cnt[:, None]
    return log_softmax(np.where(ok, logits_raw - mean[n] + bias, logits_raw), axis=1)
def native(groups):
    out = np.zeros((len(d), 18))
    for f in range(5):
        v = np.where(fq == f)[0]; c = cal[f]; pl, ce = zq['plain'][v], zq['centered'][v]
        n = np.unique(groups[v], return_inverse=True)[1]; cnt = np.bincount(n); ok = ((cnt[n] >= 2) & (cnt[n] <= 18))[:, None]
        mean = np.zeros((len(cnt), 18)); np.add.at(mean, n, ce); mean /= cnt[:, None]
        a = log_softmax(pl * c['plain_scale'], axis=1); b = log_softmax((ce - mean[n]) * c['centered_scale'], axis=1); w = np.where(ok, c['selected_centered_weight'], 0)
        out[v] = log_softmax((1 - w) * a + w * b, axis=1)
    return out
Nn = native(est); N8c = centre_lp(C8, B8, est); N8p = log_softmax(P8, axis=1)
z = lambda a: a / a.std()
print('est groups: %d clusters (true %d)' % (est.max() + 1, len(np.unique(g))))
print('native expert (est groups)        %.4f' % rec(y, Nn.argmax(1)))
print('q8v5 plain                         %.4f' % rec(y, N8p.argmax(1)))
print('q8v5 centred head (est groups)     %.4f' % rec(y, N8c.argmax(1)))
for wc in (0.5, 0.75): print('q8v5 plain/centred mix %.2f         %.4f' % (wc, rec(y, ((1 - wc) * N8p + wc * N8c).argmax(1))))
print('prediction agreement native vs q8v5-centred: %.1f%%' % (100 * (Nn.argmax(1) == N8c.argmax(1)).mean()))
keys = {p: np.where(g == p)[0] for p in np.unique(g)}; kl = list(keys); rng = np.random.default_rng(0)
boots = [np.concatenate([keys[k] for k in rng.choice(kl, len(kl))]) for _ in range(300)]
for name, M in (('q8v5 centred', N8c), ('q8v5 0.75 centred', 0.25 * N8p + 0.75 * N8c)):
    print('native + %s:' % name)
    for s in (0.3, 0.4, 0.5):
        S = (1 - s) * z(Nn) + s * z(M); dl = [rec(y[b], S[b].argmax(1)) - rec(y[b], Nn[b].argmax(1)) for b in boots]
        print('   share %.1f  %.4f   vs native alone %+.3f (%+.3f to %+.3f)' % (s, rec(y, S.argmax(1)), np.mean(dl), np.percentile(dl, 2.5), np.percentile(dl, 97.5)))

# ---- three-way check on the 539 rows stage-2 Gemma held out (30 prompts) ----
prompts_all = np.array(sorted(d.prompt_id.unique())); r0 = np.random.default_rng(0); vp = set(r0.choice(prompts_all, 30, replace=False)); isv = d.prompt_id.isin(vp).to_numpy(); idx = np.where(isv)[0]
G = np.load('kaggle_results_stage2_pred/stage2/runs/fold/pred.npz')['logp'].astype(np.float64); assert len(G) == len(idx)
yv = y[idx]; keysv = {p: np.where(g[idx] == p)[0] for p in np.unique(g[idx])}; klv = list(keysv); rv = np.random.default_rng(1)
bv = [np.concatenate([keysv[k] for k in rv.choice(klv, len(klv))]) for _ in range(300)]
print('\n--- 539 held-out rows: Gemma alone %.4f | native %.4f | q8v5-centred %.4f' % (rec(yv, G.argmax(1)), rec(yv, Nn[idx].argmax(1)), rec(yv, N8c[idx].argmax(1))))
base = 0.35 * z(G) + 0.65 * z(Nn[idx]); b0 = [rec(yv[b], base[b].argmax(1)) for b in bv]
print('current recipe (Gemma 0.35 / native 0.65): %.4f' % rec(yv, base.argmax(1)))
for wg, wn, w8 in ((0.35, 0.325, 0.325), (0.25, 0.375, 0.375), (0.2, 0.4, 0.4), (0.15, 0.425, 0.425), (0.25, 0.5, 0.25), (0.25, 0.25, 0.5)):
    S = wg * z(G) + wn * z(Nn[idx]) + w8 * z(N8c[idx]); dl = [rec(yv[b], S[b].argmax(1)) - x for b, x in zip(bv, b0)]
    print('Gemma %.2f native %.3f q8v5 %.3f : %.4f   vs current recipe %+.3f (%+.3f to %+.3f)' % (wg, wn, w8, rec(yv, S.argmax(1)), np.mean(dl), np.percentile(dl, 2.5), np.percentile(dl, 97.5)))
