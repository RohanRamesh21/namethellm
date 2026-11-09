"""Do style embedders add anything to the n-gram + Qwen-embedding + BERT blend?  (prompt-grouped CV on the official train rows)

Usage: python3 style_embedders.py --data DIR_WITH_train.csv --out RESULTS_DIR [--models CACHE_DIR]
                                  [--candidates a,b,c] [--local key=PATH ...]

Everything is rebuilt here so the comparison is like for like:
  baseline blend = n-gram scores centred per prompt + Qwen3-Embedding (middle layer, centred) + BERT (middle layer, centred)
  candidate      = same blend plus one style embedder (its best layer by centred standalone score, so a little optimistic)
Features are mean-pooled hidden states. "Centred" means each prompt group's mean feature vector is subtracted first (true
prompt groups, as in the earlier local tests). Logistic regression, C = 0.01, 5 folds that keep whole prompts together.
Writes results.md / results.json, the pooled embeddings (float16) and each model's out-of-fold scores to RESULTS_DIR.
"""
import argparse, json, shutil, sys, time, traceback, urllib.request
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer

BASELINES = {'qwen_emb': 'Qwen/Qwen3-Embedding-0.6B', 'bert': 'google-bert/bert-base-cased'}
CANDIDATES = {
    'styledistance': 'StyleDistance/styledistance',                          # roberta-base, MIT
    'wegmann': 'AnnaWegmann/Style-Embedding',                                # roberta-base, content-independent style
    'luar_mud': 'gabrielloiseau/LUAR-MUD-sentence-transformers',             # community conversion of LUAR, 6 layers
    'luar_crud': 'gabrielloiseau/LUAR-CRUD-sentence-transformers',
    'star': 'AIDA-UPM/star',                                                 # roberta-large, 24 layers
}
LAYER_FRACTIONS = (0.5, 0.75, 1.0)   # which depths to pool: middle layers beat the last layer for BERT and Qwen
C = 0.01

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True); ap.add_argument('--out', required=True); ap.add_argument('--models', default='/tmp/models')
ap.add_argument('--candidates', default=','.join(CANDIDATES)); ap.add_argument('--local', action='append', default=[])
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
local = dict(x.split('=', 1) for x in args.local)
T0 = time.time()
def log(*x): print(f'[{(time.time() - T0) / 60:6.1f}m]', *x, flush=True)


def fetch(repo):
    """Plain HTTP download with a size check on every file (the same approach the training notebooks use)."""
    target = Path(args.models) / repo.replace('/', '--'); target.mkdir(parents=True, exist_ok=True)
    info = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60))
    files = {s['rfilename']: s.get('size') for s in info['siblings'] if '/' not in s['rfilename']}
    has_safetensors = any(n.endswith('.safetensors') for n in files)
    for name, size in files.items():
        if not (name.endswith(('.json', '.txt', '.model', '.safetensors')) or (name.endswith('.bin') and not has_safetensors)): continue
        path = target / name
        if path.exists() and path.stat().st_size == size: continue
        with urllib.request.urlopen(f'https://huggingface.co/{repo}/resolve/main/{name}', timeout=120) as source, open(path, 'wb') as sink:
            shutil.copyfileobj(source, sink, 1 << 24)
        assert size is None or path.stat().st_size == size, f'{name} is incomplete'
    return str(target)


def embed(key, path, texts, fractions):
    """Mean-pooled hidden states at chosen depths. Cached on disk so a rerun skips finished models."""
    cache = out / f'emb_{key}.npz'
    if cache.exists():
        z = np.load(cache); log(f'{key}: embeddings loaded from {cache.name}'); return z['emb'].astype(np.float32), [int(l) for l in z['layers']]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).float().to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    n_layers = model.config.num_hidden_layers; layers = sorted({max(1, round(f * n_layers)) for f in fractions})
    log(f'{key}: {n_layers} layers, pooling layers {layers}, hidden {model.config.hidden_size}')
    order = np.argsort([len(t) for t in texts]); res = np.zeros((len(texts), len(layers), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(order), 16):
            idx = order[i:i + 16]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=512, return_tensors='pt').to(model.device)
            hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc['attention_mask'].unsqueeze(-1).float()
            for li, L in enumerate(layers): res[idx, li] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
    del model; torch.cuda.empty_cache()
    np.savez(cache, emb=res.astype(np.float16), layers=np.array(layers))
    return res, layers


def center_by_group(X, groups):
    X = np.asarray(X, dtype=np.float64).copy()
    for g in np.unique(groups):
        m = groups == g; X[m] -= X[m].mean(0, keepdims=True)
    return X


def macro_recall(y, p): return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def lr_oof(X, centre):
    oof = np.zeros((len(y), len(classes)))
    for tr, te in splits:
        Xtr, Xte = X[tr], X[te]
        if centre: Xtr, Xte = center_by_group(Xtr, groups[tr]), center_by_group(Xte, groups[te])
        sc = StandardScaler().fit(Xtr)
        oof[te] = LogisticRegression(C=C, max_iter=400).fit(sc.transform(Xtr), y[tr]).decision_function(sc.transform(Xte))
    return oof


def paired_delta(a, b, reps=1000):
    """Score of a minus score of b, resampling whole prompts: mean and 95% interval."""
    rows = {p: np.where(groups == p)[0] for p in np.unique(groups)}; keys = list(rows); rng = np.random.default_rng(0); d = []
    for _ in range(reps):
        r = np.concatenate([rows[k] for k in rng.choice(keys, len(keys))])
        d.append(macro_recall(y[r], a[r]) - macro_recall(y[r], b[r]))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


z = lambda a: a / a.std()
data_file = Path(args.data) / 'train.csv'
d = pd.read_csv(data_file); d['text'] = d.text.fillna('').astype(str)
classes = sorted(d.model.unique()); y = d.model.map({c: i for i, c in enumerate(classes)}).to_numpy(); groups = d.prompt_id.to_numpy()
texts = d.text.tolist(); splits = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
log(f'{len(d)} rows, {len(np.unique(groups))} prompts, {len(classes)} models; candidates: {args.candidates}')

results = {'baseline': {}, 'candidates': {}}
def write_results():
    (out / 'results.json').write_text(json.dumps(results, indent=1))
    b = results['baseline']; L = ['## Baseline (prompt-grouped 5-fold balanced accuracy)', '', '| Model | Raw | Centred per prompt |', '|---|---|---|']
    for k, v in b.get('models', {}).items(): L.append(f"| {k} | {v['raw']:.3f} | {v['centred']:.3f} |")
    if 'blend_centred' in b: L += ['', f"Blend (n-gram + Qwen-emb + BERT, all centred): **{b['blend_centred']:.3f}**. Same blend, nothing centred: {b['blend_raw']:.3f}."]
    L += ['', '## Style embedders', '', '| Model | Layers pooled | Best layer | Raw | Centred | Blend with it | Change vs baseline blend (95% interval over prompts) |', '|---|---|---|---|---|---|---|']
    for k, v in results['candidates'].items():
        if 'error' in v: L.append(f"| {k} | | | | | FAILED: {v['error'][:90]} | |"); continue
        L.append(f"| {k} | {v['layers']} | {v['best_layer']} | {v['raw']:.3f} | {v['centred']:.3f} | {v['blend']:.3f} | {v['delta'][0]:+.3f} ({v['delta'][1]:+.3f} to {v['delta'][2]:+.3f}) |")
    if 'all_added' in results:
        a = results['all_added']; L += ['', f"All working style embedders added together: {a['blend']:.3f}, change {a['delta'][0]:+.3f} ({a['delta'][1]:+.3f} to {a['delta'][2]:+.3f})."]
    L += ['', 'Best layer is picked on the same rows it is scored on, so each candidate row is slightly optimistic. An interval that includes zero means no detectable gain.']
    (out / 'results.md').write_text('\n'.join(L) + '\n')


# --- baseline: n-gram scores
word = lambda: TfidfVectorizer(lowercase=False, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
char = lambda: TfidfVectorizer(lowercase=False, analyzer='char_wb', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=300_000)
oof_ng = np.zeros((len(y), len(classes)))
for tr, te in splits:
    w, c = word(), char(); t = np.array(texts, dtype=object)
    Xtr = sp.hstack([w.fit_transform(t[tr]), c.fit_transform(t[tr])]).tocsr(); Xte = sp.hstack([w.transform(t[te]), c.transform(t[te])]).tocsr()
    oof_ng[te] = LinearSVC(C=0.5).fit(Xtr, y[tr]).decision_function(Xte)
ng_c = center_by_group(oof_ng, groups)
results['baseline']['models'] = {'ngram': {'raw': macro_recall(y, oof_ng.argmax(1)), 'centred': macro_recall(y, ng_c.argmax(1))}}
log('n-gram done', results['baseline']['models']['ngram'])

# --- baseline: frozen embedders
base_c, base_raw = {}, {}
for key, repo in BASELINES.items():
    E, layers = embed(key, local.get(key) or fetch(repo), texts, (0.5,))
    base_raw[key], base_c[key] = lr_oof(E[:, 0], False), lr_oof(E[:, 0], True)
    results['baseline']['models'][key] = {'raw': macro_recall(y, base_raw[key].argmax(1)), 'centred': macro_recall(y, base_c[key].argmax(1)), 'layer': layers[0]}
    log(key, results['baseline']['models'][key])
blend_c = z(ng_c) + z(base_c['qwen_emb']) + z(base_c['bert'])
blend_raw = z(oof_ng) + z(base_raw['qwen_emb']) + z(base_raw['bert'])
results['baseline']['blend_centred'] = macro_recall(y, blend_c.argmax(1)); results['baseline']['blend_raw'] = macro_recall(y, blend_raw.argmax(1))
log('baseline blend', results['baseline']['blend_centred'], 'raw', results['baseline']['blend_raw']); write_results()

# --- style embedders
chosen = {}
for key in [k for k in args.candidates.split(',') if k]:
    try:
        E, layers = embed(key, local.get(key) or fetch(CANDIDATES[key]), texts, LAYER_FRACTIONS)
        per = {}
        for li, L in enumerate(layers):
            raw, cen = lr_oof(E[:, li], False), lr_oof(E[:, li], True)
            per[L] = (macro_recall(y, raw.argmax(1)), macro_recall(y, cen.argmax(1)), cen)
            log(f'{key} layer {L}: raw {per[L][0]:.3f} centred {per[L][1]:.3f}')
        best = max(per, key=lambda L: per[L][1]); cen = per[best][2]; chosen[key] = cen
        np.save(out / f'oof_{key}_L{best}_centred.npy', cen.astype(np.float32))
        blend = blend_c + z(cen)
        results['candidates'][key] = {'layers': layers, 'best_layer': best, 'raw': per[best][0], 'centred': per[best][1],
                                      'blend': macro_recall(y, blend.argmax(1)), 'delta': paired_delta(blend.argmax(1), blend_c.argmax(1)),
                                      'by_layer': {str(L): {'raw': v[0], 'centred': v[1]} for L, v in per.items()}}
        log(key, 'blend', results['candidates'][key]['blend'], 'delta', results['candidates'][key]['delta'])
    except Exception:
        results['candidates'][key] = {'error': traceback.format_exc().strip().splitlines()[-1]}; log(key, 'FAILED', results['candidates'][key]['error'])
    write_results()

if len(chosen) > 1:
    allb = blend_c + sum(z(c) for c in chosen.values())
    results['all_added'] = {'blend': macro_recall(y, allb.argmax(1)), 'delta': paired_delta(allb.argmax(1), blend_c.argmax(1))}
write_results(); log('finished'); print((out / 'results.md').read_text())
