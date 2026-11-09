"""Name That LLM: offline submission pipeline.

Recipe (validated in experiments/sim_unequal.py): 2 x Gemma LoRA scores (raw) + n-gram scores (centred per prompt group)
+ Qwen3-Embedding mid-layer LR (centred) + BERT mid-layer LR (centred). "Centred" means each prompt group's mean is subtracted.
The test set has no prompt ids, so groups are estimated from the text: Ward clustering of Qwen last-layer features, with the
number of clusters chosen by silhouette (on training rows it lands within 0.91-1.08x of the true prompt count).

run_pipeline() takes the Gemma scorer as a function so everything else can be tested without Unsloth.
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.special import log_softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC

WEIGHTS = {'gemma': 2.0, 'ngram': 1.0, 'qwen': 1.0, 'bert': 1.0}


def macro_recall(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def center_by_group(X, groups):
    X = np.asarray(X, dtype=np.float64).copy()
    for g in np.unique(groups):
        m = groups == g
        X[m] -= X[m].mean(0, keepdims=True)
    return X


def assign_by_group(S, groups):
    """One model per passage inside each group, each model used at most ceil(size/18) times. Only valid if prompts really
    hold one passage per model, which the competition says is not guaranteed. Off by default."""
    pred = S.argmax(1).copy(); k = S.shape[1]
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        r, c = linear_sum_assignment(-np.tile(S[idx], (1, int(np.ceil(len(idx) / k)))))
        pred[idx[r]] = c % k
    return pred


def estimate_groups(X, lo=40, hi=4, n_grid=24):
    """Cluster passages into prompt groups from text features alone. Returns (labels, info)."""
    N = len(X)
    if N < 30:
        return np.zeros(N, dtype=int), {'K': 1, 'note': f'only {N} rows: no grouping'}
    Xn = normalize(X - X.mean(0))
    Z = linkage(Xn, 'ward'); D = pairwise_distances(Xn)
    best = None
    for K in sorted({int(round(k)) for k in np.geomspace(max(2, N / lo), max(3, N / hi), n_grid)}):
        lab = fcluster(Z, K, 'maxclust')
        if not 1 < len(np.unique(lab)) < N: continue
        s = silhouette_score(D, lab, metric='precomputed')
        if best is None or s > best[1]: best = (K, s, lab)
    if best is None:
        return np.zeros(N, dtype=int), {'K': 1, 'note': 'no usable clustering'}
    K, s, lab = best
    lab = np.unique(lab, return_inverse=True)[1]; sizes = np.bincount(lab)
    return lab, {'K': int(len(sizes)), 'silhouette': float(s), 'size_min': int(sizes.min()), 'size_median': float(np.median(sizes)),
                 'size_mean': float(sizes.mean()), 'size_max': int(sizes.max()), 'singletons': int((sizes == 1).sum())}


def ngram_scores(pool, labels, text_sets, log=print, extra=None, extra_weight=1.0):
    """The same n-gram SVC the training notebook scored the official rows with, trained on the pool rows (plus, optionally, official rows
    at extra_weight: extra = (texts, labels))."""
    t = time.time()
    svc = Pipeline([('f', FeatureUnion([
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=300000, sublinear_tf=True, lowercase=False, dtype=np.float32)),
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=100000, sublinear_tf=True, lowercase=False, token_pattern=r'\w+|[^\w\s]', dtype=np.float32))])),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', max_iter=5000))])
    X, Y = pool.text.tolist(), pool.model.tolist(); sw = np.ones(len(X))
    if extra is not None:
        X += list(extra[0]); Y += list(extra[1]); sw = np.r_[sw, np.full(len(extra[0]), float(extra_weight))]
    svc.fit(X, Y, clf__sample_weight=sw)
    cols = [list(svc.classes_).index(l) for l in labels]
    log(f'n-gram fitted on {len(X)} rows ({len(pool)} pool + {len(X) - len(pool)} official) in {(time.time() - t) / 60:.1f} min')
    return [log_softmax(svc.decision_function(texts)[:, cols] * 4, axis=1) if len(texts) else np.zeros((0, len(labels))) for texts in text_sets]


def embed(path, texts, depths, log=print, batch=16, max_len=512):
    """Mean-pooled hidden states. depths are fractions of the layer count (0.5 = middle, 1.0 = last). Returns {layer: array}."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModel.from_pretrained(path).float().to(dev).eval()
    n = model.config.num_hidden_layers; layers = sorted({max(1, round(f * n)) for f in depths})
    texts = list(texts); order = np.argsort([len(t) for t in texts]); out = {L: np.zeros((len(texts), model.config.hidden_size), np.float32) for L in layers}
    t = time.time()
    with torch.no_grad():
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            enc = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states
            m = enc['attention_mask'].unsqueeze(-1).float()
            for L in layers: out[L][idx] = ((hs[L].float() * m).sum(1) / m.sum(1)).cpu().numpy()
    del model
    if dev == 'cuda': torch.cuda.empty_cache()
    log(f'embedded {len(texts)} passages with {Path(path).name} ({n} layers, pooled {layers}) in {(time.time() - t) / 60:.1f} min')
    return out


def frozen_scores(E_train, y_train, g_train, E_apply, g_apply, C=0.01, center=True):
    """Logistic regression on prompt-centred features. Training rows are centred by their true prompt, rows to score by their estimated group."""
    Xtr = center_by_group(E_train, g_train) if center else np.asarray(E_train, dtype=np.float64); sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, max_iter=400).fit(sc.transform(Xtr), y_train)
    return clf.decision_function(sc.transform(center_by_group(E_apply, g_apply) if center else np.asarray(E_apply, dtype=np.float64)))


def blend(parts, weights=WEIGHTS):
    return sum(weights[k] * (v / v.std()) for k, v in parts.items())


def run_pipeline(comp_dir, pool_path, qwen_dir, bert_dir, gemma_fn, out_dir, validate=True, val_prompts=40, require_gemma=True,
                 use_assignment=False, holdout_prompts=(), test_df=None, pool_limit=None, variant='recipe', ngram_official_weight=0.0, log=print):
    """variant: 'recipe' (default), 'gemma_only' (Gemma scores alone, to see what it really scores on the test) or 'no_centering'."""
    t0 = time.time(); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(f'{comp_dir}/train.csv'); train['text'] = train.text.fillna('').astype(str)
    test = test_df.copy() if test_df is not None else pd.read_csv(f'{comp_dir}/test.csv'); test['text'] = test.text.fillna('').astype(str)
    labels = sorted(train.model.unique()); y = train.model.map({l: i for i, l in enumerate(labels)}).to_numpy(); g_true = train.prompt_id.to_numpy()
    usable = ~np.isin(g_true, list(holdout_prompts))   # rows allowed to train the frozen models (everything, except in local tests)
    rng = np.random.default_rng(0)
    val_set = set(rng.choice(np.unique(g_true[usable]), val_prompts, replace=False)) if validate else set()
    val_idx = np.where(np.isin(g_true, list(val_set)) & usable)[0]
    log(f'{len(train)} train rows, {len(test)} test rows; validation on {len(val_idx)} held-out train rows ({len(val_set)} prompts)')

    # 1. Gemma LoRA scores for the validation rows and the test rows in one call
    texts = list(train.text.iloc[val_idx]) + list(test.text)
    try:
        G = np.asarray(gemma_fn(texts), dtype=np.float64); gemma_ok = True
    except Exception as error:
        if require_gemma: raise
        log(f'!!! GEMMA FAILED, continuing without it: {error!r}'); G = np.zeros((len(texts), len(labels))); gemma_ok = False
    G_val, G_test = G[:len(val_idx)], G[len(val_idx):]

    # 2. n-gram scores from the pool rows
    pool = pd.read_parquet(pool_path, columns=['text', 'model'])
    if pool_limit: pool = pool.sample(min(pool_limit, len(pool)), random_state=0)
    extra = None
    if ngram_official_weight > 0:   # official rows join the n-gram training, except validation rows so validation stays honest
        keep = usable.copy(); keep[val_idx] = False
        extra = (list(train.text[keep]), list(train.model[keep]))
    N_val, N_test = ngram_scores(pool, labels, [train.text.iloc[val_idx], test.text], log, extra=extra, extra_weight=ngram_official_weight)

    # 3. frozen embedders on all training rows and on the test rows
    q_tr = embed(qwen_dir, train.text, (0.5, 1.0), log); q_te = embed(qwen_dir, test.text, (0.5, 1.0), log)
    b_tr = embed(bert_dir, train.text, (0.5,), log); b_te = embed(bert_dir, test.text, (0.5,), log)
    qmid, qlast = sorted(q_tr)[0], sorted(q_tr)[-1]; bmid = sorted(b_tr)[0]

    def recipe(G_, N_, qs_tr_mask, q_apply, b_apply, last_apply, true_labels=None):
        est, info = estimate_groups(last_apply)
        if variant == 'gemma_only':
            parts = {'gemma': G_}
        else:
            c = variant != 'no_centering'
            fq = frozen_scores(q_tr[qmid][qs_tr_mask], y[qs_tr_mask], g_true[qs_tr_mask], q_apply, est, center=c)
            fb = frozen_scores(b_tr[bmid][qs_tr_mask], y[qs_tr_mask], g_true[qs_tr_mask], b_apply, est, center=c)
            parts = {'gemma': G_, 'ngram': center_by_group(N_, est) if c else N_, 'qwen': fq, 'bert': fb}
        if not gemma_ok: parts.pop('gemma', None)
        S = blend(parts, {k: WEIGHTS[k] for k in parts})
        pred = assign_by_group(S, est) if use_assignment else S.argmax(1)
        return S, pred, parts, est, info

    report = {'weights': WEIGHTS, 'gemma_used': gemma_ok, 'use_assignment': bool(use_assignment), 'variant': variant, 'ngram_official_weight': ngram_official_weight}
    # 4. validation: frozen models see only training prompts outside the validation prompts
    if validate and len(val_idx):
        mask = usable.copy(); mask[val_idx] = False
        S, pred, parts, est, info = recipe(G_val, N_val, mask, q_tr[qmid][val_idx], b_tr[bmid][val_idx], q_tr[qlast][val_idx])
        yv = y[val_idx]
        report['validation'] = {'rows': int(len(val_idx)), 'groups': info, 'recipe': macro_recall(yv, pred),
                                'components': {k: macro_recall(yv, v.argmax(1)) for k, v in parts.items()}}
        log('validation (held-out training prompts):', json.dumps(report['validation']))
        report['_val_logp'] = (val_idx, G_val)
    # 5. test: frozen models see every usable training row
    S, pred, parts, est, info = recipe(G_test, N_test, usable, q_te[qmid], b_te[bmid], q_te[qlast])
    report['test_groups'] = info; log('test groups:', json.dumps(info))
    np.savez(out_dir / 'parent_scores.npz', ids=test['id'].astype(str).to_numpy(dtype=str), S=S, groups=est, labels=np.asarray(labels))   # for blending with other models later
    sub = pd.DataFrame({'id': test['id'].astype(str), 'model': [labels[i] for i in pred]})
    assert sub.id.is_unique and sub.model.notna().all() and set(sub.model) <= set(labels), 'invalid submission'
    sub.to_parquet(out_dir / 'submission.parquet', index=False)
    report['test_prediction_counts'] = sub.model.value_counts().to_dict(); report['minutes'] = (time.time() - t0) / 60
    (out_dir / 'submission_info.json').write_text(json.dumps({k: v for k, v in report.items() if not k.startswith('_')}, indent=1))
    log(f'wrote {out_dir / "submission.parquet"} ({len(sub)} rows) in {report["minutes"]:.1f} min')
    report['_pred'] = pred; report['_labels'] = labels
    return report
