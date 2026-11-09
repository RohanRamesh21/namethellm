"""Out-of-fold check of the 14B L24 expert (q14v5) alone and together with the native expert and q8v5, with estimated prompt groups."""
import json, sys
import numpy as np, pandas as pd
from scipy.special import log_softmax
import importlib.util
spec = importlib.util.spec_from_file_location('P', 'experiments/ntl_pipeline.py'); P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
d = pd.read_csv('name-that-llm/train.csv'); classes = sorted(d.model.unique()); y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy(); g = d.prompt_id.to_numpy()
prompts = np.unique(g); rows = np.arange(len(y))
rec = lambda yy, p: float(np.mean([(p[yy == c] == c).mean() for c in np.unique(yy)]))
def oof(path):
    z = np.load(path); Pl = z['plain'].astype(np.float64)
    lp = np.stack([log_softmax(Pl[a], axis=1)[rows, y] for a in range(Pl.shape[0])])
    loss = np.array([[-lp[a][g == p].mean() for p in prompts] for a in range(Pl.shape[0])]); held = loss.argmax(0)
    srt = np.sort(loss, axis=0); fold = np.array([dict(zip(prompts, held))[p] for p in g])
    return dict(plain=Pl[fold, rows], craw=z['centred_raw'].astype(np.float64)[fold, rows], bias=z['centred_bias'][fold], fold=fold,
                per_fold=np.bincount(held, minlength=5), gap=float(np.median(srt[-1] - srt[-2])))
A8 = oof('kaggle_results_l24/q8v5_official.npz'); A14 = oof('kaggle_results_l24_q14/q14v5_official.npz')
print('q14v5 prompts per held-out adapter', A14['per_fold'], '| median loss gap worst vs second-worst %.2f' % A14['gap'])
np.save('experiments/cache/q14v5_fold_of_row.npy', A14['fold'])
Q = 'public_assets/native_q8'; zq = np.load(f'{Q}/oof_reference.npz'); fq = np.zeros(len(d), int)
for f in range(5): fq[json.load(open(f'{Q}/fold{f}/split.json'))['validation_rows']] = f
cal = [json.load(open(f'{Q}/calibration/fold{f}.json'))['models']['qwen8'] for f in range(5)]
q = np.load('experiments/cache/emb_qwen.npy')[:, 1]
est = np.zeros(len(d), int); off = 0
for f in range(5):
    v = np.where(fq == f)[0]; lab, _ = P.estimate_groups(q[v]); est[v] = lab + off; off += lab.max() + 1
def inv_cnt(groups):
    n = np.unique(groups, return_inverse=True)[1]; cnt = np.bincount(n); return n, cnt, ((cnt[n] >= 2) & (cnt[n] <= 18))[:, None]
def l24_lp(A, groups):
    n, cnt, ok = inv_cnt(groups); mean = np.zeros((len(cnt), 18)); np.add.at(mean, n, A['craw']); mean /= cnt[:, None]
    return log_softmax(np.where(ok, A['craw'] - mean[n] + A['bias'], A['plain']), axis=1)
def native(groups):
    out = np.zeros((len(d), 18))
    for f in range(5):
        v = np.where(fq == f)[0]; c = cal[f]; pl, ce = zq['plain'][v], zq['centered'][v]; n, cnt, ok = inv_cnt(groups[v])
        mean = np.zeros((len(cnt), 18)); np.add.at(mean, n, ce); mean /= cnt[:, None]
        a = log_softmax(pl * c['plain_scale'], axis=1); b = log_softmax((ce - mean[n]) * c['centered_scale'], axis=1); w = np.where(ok, c['selected_centered_weight'], 0)
        out[v] = log_softmax((1 - w) * a + w * b, axis=1)
    return out
Nn, N8, N14 = native(est), l24_lp(A8, est), l24_lp(A14, est)
z = lambda a: a / a.std()
print('est groups: %d clusters' % (est.max() + 1))
print('native %.4f | q8v5 %.4f | q14v5 plain %.4f centred %.4f' % (rec(y, Nn.argmax(1)), rec(y, N8.argmax(1)), rec(y, log_softmax(A14['plain'], axis=1).argmax(1)), rec(y, N14.argmax(1))))
print('agreement: native-q14 %.1f%%  q8-q14 %.1f%%  native-q8 %.1f%%' % tuple(100 * (a.argmax(1) == b.argmax(1)).mean() for a, b in ((Nn, N14), (N8, N14), (Nn, N8))))
keys = {p: np.where(g == p)[0] for p in prompts}; kl = list(keys); rng = np.random.default_rng(0); boots = [np.concatenate([keys[k] for k in rng.choice(kl, len(kl))]) for _ in range(300)]
def show(name, S, ref):
    dl = [rec(y[b], S[b].argmax(1)) - rec(y[b], ref[b].argmax(1)) for b in boots]
    print('  %-34s %.4f  vs %-12s %+.3f (%+.3f to %+.3f)' % (name, rec(y, S.argmax(1)), 'previous', np.mean(dl), np.percentile(dl, 2.5), np.percentile(dl, 97.5)))
base2 = 0.5 * z(Nn) + 0.5 * z(N8)
print('native + q14v5 (vs native alone):')
for s in (0.4, 0.5): show('native %.1f + q14v5 %.1f' % (1 - s, s), (1 - s) * z(Nn) + s * z(N14), Nn)
print('three experts (vs native + q8v5 50/50 = %.4f):' % rec(y, base2.argmax(1)))
for wn, w8, w14 in ((1/3, 1/3, 1/3), (0.4, 0.3, 0.3), (0.5, 0.25, 0.25), (0.3, 0.2, 0.5), (0.4, 0.2, 0.4)):
    show('native %.2f q8v5 %.2f q14v5 %.2f' % (wn, w8, w14), wn * z(Nn) + w8 * z(N8) + w14 * z(N14), base2)
# same on the 539 stage-2 held-out rows with Gemma
prompts_all = np.array(sorted(prompts)); r0 = np.random.default_rng(0); vp = set(r0.choice(prompts_all, 30, replace=False)); idx = np.where(d.prompt_id.isin(vp).to_numpy())[0]
G = np.load('kaggle_results_stage2_pred/stage2/runs/fold/pred.npz')['logp'].astype(np.float64); yv = y[idx]
print('\n539 held-out rows (Gemma stage 2 %.4f, native %.4f, q8v5 %.4f, q14v5 %.4f):' % tuple(rec(yv, a[idx].argmax(1) if a is not G else G.argmax(1)) for a in (G, Nn, N8, N14)))
for name, S in (('Gemma .35 native .65 (v5)', .35 * z(G) + .65 * z(Nn[idx])), ('Gemma .30 native .35 q8 .35 (l24 nb)', .30 * z(G) + .35 * z(Nn[idx]) + .35 * z(N8[idx])),
                ('Gemma .30 native .25 q8 .20 q14 .25', .30 * z(G) + .25 * z(Nn[idx]) + .20 * z(N8[idx]) + .25 * z(N14[idx])), ('Gemma .20 native .30 q8 .20 q14 .30', .20 * z(G) + .30 * z(Nn[idx]) + .20 * z(N8[idx]) + .30 * z(N14[idx])),
                ('Gemma .25 native .25 q8 .25 q14 .25', .25 * z(G) + .25 * z(Nn[idx]) + .25 * z(N8[idx]) + .25 * z(N14[idx]))):
    print('  %-38s %.4f' % (name, rec(yv, S.argmax(1))))
