"""Builds notebooks/l24_probe.ipynb: runs the q8v5 fold adapters over all 2,699 official rows (for clean out-of-fold checks)."""
import argparse, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
ap = argparse.ArgumentParser(); ap.add_argument("--name", default="q8v5"); ap.add_argument("--model", default="qwen-lm/qwen-3/transformers/8b-base/1"); ap.add_argument("--keep", type=int, default=24)
ap.add_argument("--bs", type=int, default=16); ap.add_argument("--out", default="/home/rohan/kagglecomp/notebooks/l24_probe.ipynb")
args = ap.parse_args()
MODEL_PATHS = ["/kaggle/input/models/" + args.model, "/kaggle/input/" + args.model.split("/", 1)[1]]
worker = open('/home/rohan/kagglecomp/experiments/l24_worker.py').read()
cells = [
 new_markdown_cell("""# L24 expert probe (q8v5)

Runs the public Qwen3-8B-Base 'L24' LoRA expert (dhruvtibarewal/nlm-l24-adapters-v5, q8v5) with my own worker over all 2,699 official rows, once per fold adapter (f0-f4),
so each row has one score from an adapter that never saw its prompt. The held-out fold of each adapter is worked out afterwards from the accuracy pattern.

Inputs: the **Name That LLM** data, the dataset `dhruvtibarewal/nlm-l24-adapters-v5`, the Kaggle model `qwen-lm/qwen-3` (transformers, 8b-base). GPU T4 x2."""),
 new_code_cell("""import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

def first(paths, what):
    for p in paths:
        if os.path.exists(p): return p
    raise SystemExit(f'Could not find {what}. Tried {paths}. /kaggle/input has: {sorted(os.listdir("/kaggle/input"))}')
COMP = first(['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'], 'the competition data')
ADAPTERS = first(['/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5', '/kaggle/input/nlm-l24-adapters-v5'], 'the L24 adapters dataset')
BASE = first(""" + repr(MODEL_PATHS) + """, 'the base model')
print('COMP', COMP); print('ADAPTERS', ADAPTERS, sorted(os.listdir(ADAPTERS))[:12]); print('BASE', BASE, sorted(os.listdir(BASE)))
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__, '| gpus', torch.cuda.device_count())"""),
 new_code_cell('%%writefile l24_worker.py\n' + worker.rstrip('\n')),
 new_code_cell("""off = pd.read_csv(f'{COMP}/train.csv'); off['text'] = off.text.fillna('').astype(str)
pd.DataFrame({'id': [str(i) for i in range(len(off))], 'text': off.text.values}).to_csv('/kaggle/working/query.csv', index=False)
t = time.time()
p = subprocess.Popen([sys.executable, '-u', 'l24_worker.py', '--base', BASE, '--adapters', ','.join(f'{ADAPTERS}/NAME_f{k}' for k in range(5)), '--query', '/kaggle/working/query.csv',
                      '--out', '/kaggle/working/NAME_official.npz', '--keep', 'KEEP', '--bs', 'BS'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                     env=dict(os.environ, TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1'))
tail = []
for line in p.stdout:
    print(line, end=''); tail = (tail + [line])[-40:]
if p.wait(): raise SystemExit('worker failed:\\n' + ''.join(tail[-25:]))
print(f'worker wall time {(time.time() - t) / 60:.1f} min')"""),
 new_code_cell("""try:   # quick look only; the real analysis happens locally
    from sklearn.model_selection import GroupKFold
    z = np.load('/kaggle/working/NAME_official.npz'); y = off.model.map({c: i for i, c in enumerate(sorted(off.model.unique()))}).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, off.prompt_id.to_numpy()))
    for a in range(z['plain'].shape[0]):
        print(z['adapters'][a], 'accuracy on each sklearn GroupKFold fold:', [round(float((z['plain'][a][te].argmax(1) == y[te]).mean()), 3) for _, te in folds])
except Exception as e:
    print('summary skipped:', repr(e))""")]
nb = new_notebook(cells=cells); nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}
for c in nb.cells: c.pop('id', None)
nb.nbformat_minor = 4
for c in nb.cells:
    if c.cell_type == 'code': c.source = c.source.replace('NAME', args.name).replace("'KEEP'", repr(str(args.keep))).replace("'BS'", repr(str(args.bs)))
nbformat.write(nb, args.out); print('ok', args.name)
