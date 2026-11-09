"""Does adding the official training rows to the n-gram model's training data help on held-out prompts?
Pool rows (a 30k sample, for RAM) alone vs pool + official rows at weight 1 and 5. 5 prompt-grouped folds on the official rows."""
import sys, time
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
sys.path.insert(0, '/home/rohan/kagglecomp/experiments')
from common import load

d, y, groups, classes = load()
texts = d.text.fillna('').astype(str).to_numpy(); lab = np.array(classes)[y]
pool = pd.read_parquet('/home/rohan/kagglecomp/kaggle_results_v3/run/train_pool.parquet', columns=['text', 'model']).sample(30000, random_state=0)

def make():
    return Pipeline([('f', FeatureUnion([
        ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), max_features=300000, sublinear_tf=True, lowercase=False, dtype=np.float32)),
        ('word', TfidfVectorizer(ngram_range=(1, 2), max_features=100000, sublinear_tf=True, lowercase=False, token_pattern=r'\w+|[^\w\s]', dtype=np.float32))])),
        ('clf', LinearSVC(C=0.5, class_weight='balanced', max_iter=5000))])

def macro_recall(yt, p): return float(np.mean([(p[yt == c] == c).mean() for c in np.unique(yt)]))
res = {'pool only': [], 'pool + official x1': [], 'pool + official x5': []}
t0 = time.time()
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(texts, y, groups)):
    for name, w in (('pool only', None), ('pool + official x1', 1.0), ('pool + official x5', 5.0)):
        X = pool.text.tolist() + ([] if w is None else list(texts[tr])); Y = pool.model.tolist() + ([] if w is None else list(lab[tr]))
        sw = np.r_[np.ones(len(pool)), np.full(0 if w is None else len(tr), w or 1.0)]
        m = make().fit(X, Y, clf__sample_weight=sw)
        pred = np.array([list(m.classes_).index(c) for c in np.array(m.predict(texts[te]))])
        pred = np.array([classes.index(m.classes_[i]) for i in [list(m.classes_).index(p) for p in m.predict(texts[te])]])
        res[name].append(macro_recall(y[te], pred))
    print(f'fold {k} done ({(time.time()-t0)/60:.1f} min):', {n: round(v[-1], 3) for n, v in res.items()}, flush=True)
print('\nmean over 5 folds:', {n: round(float(np.mean(v)), 3) for n, v in res.items()})
