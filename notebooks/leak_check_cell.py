# Leak check for gemma_vs_qwen_lora: paste as a new cell after the run finishes. CPU is enough; it takes seconds.
# Is the closest training passage to each official row written by the same model?
# Control: the same question asked of official rows from *other* prompts, where no leak is possible.
import glob, os
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

pools = [p for p in sorted(glob.glob('/kaggle/working/run/train_pool.parquet') + glob.glob('/kaggle/input/**/run/train_pool.parquet', recursive=True)) if '_smoke' not in p]
assert pools, "No train_pool.parquet found: run this in the same session, or add this notebook's output as an input."
data = next(p for p in ['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'] if os.path.exists(p + '/train.csv'))
train = pd.read_parquet(pools[0]); off = pd.read_csv(data + '/train.csv'); off['text'] = off.text.fillna('')
vec = TfidfVectorizer(sublinear_tf=True, stop_words='english', min_df=2, max_df=0.05)
X = vec.fit_transform(pd.concat([off.text, train.text])); A, B = X[:len(off)], X[len(off):]
y_off, y_pool, prompt = off.model.to_numpy(), train.model.to_numpy(), off.prompt_id.to_numpy()

def top_k_match(S, labels, truth, k):   # share of each row's k closest passages written by the same model
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return (labels[idx] == truth[:, None]).mean(1)

pool_1, pool_10, ctrl_1, ctrl_10, near = [], [], [], [], []
for s in range(0, len(off), 256):
    truth = y_off[s:s + 256]
    S = (A[s:s + 256] @ B.T).toarray(); near.append(S.max(1))
    pool_1.append(top_k_match(S, y_pool, truth, 1)); pool_10.append(top_k_match(S, y_pool, truth, 10))
    C = (A[s:s + 256] @ A.T).toarray(); C[prompt[s:s + 256, None] == prompt[None, :]] = -1   # other prompts only
    ctrl_1.append(top_k_match(C, y_off, truth, 1)); ctrl_10.append(top_k_match(C, y_off, truth, 10))
pool_1, pool_10, ctrl_1, ctrl_10, near = map(np.concatenate, (pool_1, pool_10, ctrl_1, ctrl_10, near))
print(f'closest passage by the same model   training rows: {pool_1.mean():.1%}   control: {ctrl_1.mean():.1%}   chance: {1/18:.1%}')
print(f'same model among the 10 closest     training rows: {pool_10.mean():.1%}   control: {ctrl_10.mean():.1%}')
close = near > np.quantile(near, 0.67)
print(f'top third most topic-similar rows   training rows: {pool_1[close].mean():.1%} for the closest passage')
print('\nReading it, using the first line:'
      '\n- Near the control: no sign of a leak.'
      '\n- Well above it: the training rows include the same model answering official prompts, so absolute scores are inflated.'
      '\n- Well below it: the training rows include other models answering official prompts. That is topic overlap, not a label leak.')
