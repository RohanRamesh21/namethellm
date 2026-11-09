"""Local end-to-end test of ntl_pipeline with the Gemma step replaced by the scores saved from the training run.
60 training prompts are held out as a stand-in test set (never used to train the frozen models); Gemma never saw any official row."""
import sys, glob, time, os
import numpy as np, pandas as pd
sys.path.insert(0, '/home/rohan/kagglecomp/experiments')
import ntl_pipeline as P

COMP = '/home/rohan/kagglecomp/name-that-llm'; RUN = '/home/rohan/kagglecomp/kaggle_results_v3/run'
train = pd.read_csv(f'{COMP}/train.csv'); train['text'] = train.text.fillna('').astype(str)
saved = np.load(f'{RUN}/runs/gemma_s1/pred.npz')['logp'].astype(np.float64)       # rows in train.csv order
by_text = {t: i for i, t in enumerate(train.text)}
gemma_fn = lambda texts: saved[[by_text[t] for t in texts]]

rng = np.random.default_rng(1); hold = list(rng.choice(train.prompt_id.unique(), 60, replace=False))
mask = train.prompt_id.isin(hold)
test_df = train.loc[mask, ['id', 'text']].reset_index(drop=True); truth = train.loc[mask, 'model'].to_numpy()
Q = glob.glob('/home/rohan/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/*')[0]
rep = P.run_pipeline(COMP, f'{RUN}/train_pool.parquet', Q, '/home/rohan/.cache/huggingface/manual/bert-base-cased', gemma_fn,
                     '/tmp/ntl_local_out', validate=os.environ.get('VALIDATE', '1') == '1', val_prompts=40, holdout_prompts=hold, test_df=test_df, pool_limit=20000, variant=os.environ.get('VARIANT', 'recipe'), use_assignment=os.environ.get('ASSIGN', '0') == '1', ngram_official_weight=float(os.environ.get('NGRAM_OFF', '0')))
labels = rep['_labels']; yt = np.array([labels.index(m) for m in truth])
print('\n=== pseudo-test (60 held-out prompts, %d rows): recipe balanced accuracy %.3f' % (len(yt), P.macro_recall(yt, rep['_pred'])))
print('    gemma alone on the same rows: %.3f' % P.macro_recall(yt, gemma_fn(list(test_df.text)).argmax(1)))
sub = pd.read_parquet('/tmp/ntl_local_out/submission.parquet'); print('    submission file:', sub.shape, list(sub.columns), sub.dtypes.astype(str).to_dict())
