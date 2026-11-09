"""Inference-only reader for the public pre-fitted 4-view SVC ('svc4') shipped in lightningv08/ntl-olmo2-native24-final-inference-assets
(folder ntl673-response-core/.../core/svc4: vectorizer-0..3.joblib + model.npz). Own implementation of what the public notebooks do
(views: char 3-5 weight 1.0, word 1-3 weight 0.7, word-shape 2-6 weight 0.5, function-word 1-3 weight 0.5; score = X @ coef.T + intercept).
The joblib files are loaded through experiments/safe_joblib.py (allow-listed classes only).
Trained on 231,960 rows INCLUDING the 2,699 official rows, so official-row scores are in-sample and cannot validate it.
"""
import json, re, sys
from pathlib import Path
import numpy as np
from scipy.sparse import hstack

sys.path.insert(0, str(Path(__file__).parent))
from safe_joblib import safe_load

FUNCTION_WORDS = frozenset('a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little'.split())


def shape_view(text): return re.sub(r'[^\W\d_]+', lambda m: 'X' if m[0].isupper() else 'x', text)
def function_view(text): return ' '.join(t.lower() if t.lower() in FUNCTION_WORDS or not t.isalnum() else 'CONTENT' for t in re.findall(r'\w+|[^\w\s]', text))
def prepare(texts, view): return [shape_view(t) for t in texts] if view == 'shape' else [function_view(t) for t in texts] if view == 'function' else list(texts)


class Svc4:
    def __init__(self, folder):
        folder = Path(folder); meta = json.loads((folder / 'metadata.json').read_text())
        self.labels = meta['labels']; self.views = meta['views']
        self.vectorizers = [safe_load(folder / v['file']) for v in self.views]
        with np.load(folder / 'model.npz', allow_pickle=False) as z:
            self.coef, self.intercept, classes = z['coef'].astype(np.float32), z['intercept'], z['classes']
        assert classes.tolist() == list(range(len(self.labels))) and self.coef.shape[0] == len(self.labels)

    def decision(self, texts):
        texts = [str(t) for t in texts]
        X = hstack([v.transform(prepare(texts, e['view'])) * e['weight'] for v, e in zip(self.vectorizers, self.views)], format='csr')
        return np.asarray(X @ self.coef.T) + self.intercept
