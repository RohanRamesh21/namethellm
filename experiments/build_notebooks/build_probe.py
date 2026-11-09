import json, nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
worker = open('/home/rohan/kagglecomp/experiments/native_worker.py').read()
cells = []
cells.append(new_markdown_cell("""# Native expert probe

Checks that my port of the public Qwen3-Embedding-8B expert (CHEN Han's trained fold adapters) reproduces the out-of-fold scores the author stored,
and measures how long one fold takes. It scores fold 0's 540 held-out official rows with the fold-0 adapter only.

Inputs: the **Name That LLM** competition data, the dataset `lightningv08/ntl-olmo2-native24-final-inference-assets`, and the Kaggle model `qwen-lm/qwen-3-embedding` (transformers, 8b). Settings: GPU T4 x2."""))
cells.append(new_code_cell("""import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

def first(paths, what):
    for p in paths:
        if os.path.exists(p): return p
    raise SystemExit(f'Could not find {what}. Tried {paths}. /kaggle/input has: {sorted(os.listdir("/kaggle/input"))}')

COMP = first(['/kaggle/input/competitions/name-that-llm', '/kaggle/input/name-that-llm'], 'the competition data')
ASSETS = first(['/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl-native-q8-multidepth-fivefold-0679',
                '/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets/ntl-native-q8-multidepth-fivefold-0679'], 'the native expert assets')
BASE = first(['/kaggle/input/models/qwen-lm/qwen-3-embedding/transformers/8b/1', '/kaggle/input/qwen-3-embedding/transformers/8b/1'], 'the Qwen3-Embedding-8B model')
print('COMP  ', COMP); print('ASSETS', ASSETS, sorted(os.listdir(ASSETS))); print('BASE  ', BASE, sorted(os.listdir(BASE)))
import torch, transformers
print('torch', torch.__version__, '| transformers', transformers.__version__, '| gpus', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print(subprocess.run(['df', '-h', '/kaggle/working', '/kaggle/temp'], capture_output=True, text=True).stdout)"""))
cells.append(new_code_cell('%%writefile native_worker.py\n' + worker.rstrip('\n')))
cells.append(new_code_cell("""off = pd.read_csv(f'{COMP}/train.csv'); off['text'] = off.text.fillna('').astype(str)
split = json.loads((Path(ASSETS) / 'fold0' / 'split.json').read_text())
rows = np.array(split['validation_rows']); print(len(rows), 'held-out rows of fold 0,', off.prompt_id.iloc[rows].nunique(), 'prompts')
pd.DataFrame({'id': [str(i) for i in rows], 'text': off.text.iloc[rows].values}).to_csv('/kaggle/working/probe_query.csv', index=False)
Path('/kaggle/temp/work').mkdir(parents=True, exist_ok=True)
t = time.time()
p = subprocess.Popen([sys.executable, '-u', 'native_worker.py', '--base', BASE, '--assets', ASSETS, '--query', '/kaggle/working/probe_query.csv',
                      '--out', '/kaggle/working/probe_native.npz', '--work', '/kaggle/temp/work', '--folds', '0'],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1'))
tail = []
for line in p.stdout:
    print(line, end=''); tail.append(line); tail = tail[-40:]
if p.wait(): raise SystemExit('worker failed:\\n' + ''.join(tail[-25:]))
print(f'worker wall time {(time.time() - t) / 60:.1f} min')"""))
cells.append(new_code_cell("""from scipy.special import log_softmax
z = np.load('/kaggle/working/probe_native.npz'); ref = np.load(Path(ASSETS) / 'oof_reference.npz')
y_all = off.model.map({c: i for i, c in enumerate(sorted(off.model.unique()))}).to_numpy(); y = y_all[rows]
mine_p, mine_c = z['plain'][0], z['centred'][0]; ref_p, ref_c = ref['plain'][rows], ref['centred'][rows]
def macro(p): return float(np.mean([(p[y == c] == c).mean() for c in range(18)]))
print('argmax agreement with the stored scores  plain %.1f%%  centred %.1f%%' % (100 * (mine_p.argmax(1) == ref_p.argmax(1)).mean(), 100 * (mine_c.argmax(1) == ref_c.argmax(1)).mean()))
print('max abs logit difference                 plain %.3f   centred %.3f   (logit std %.2f)' % (np.abs(mine_p - ref_p).max(), np.abs(mine_c - ref_c).max(), ref_p.std()))
print('correlation                              plain %.5f  centred %.5f' % (np.corrcoef(mine_p.ravel(), ref_p.ravel())[0, 1], np.corrcoef(mine_c.ravel(), ref_c.ravel())[0, 1]))
print('balanced accuracy of mine on these rows  plain %.4f  centred(raw) %.4f' % (macro(mine_p.argmax(1)), macro(mine_c.argmax(1))))
print('balanced accuracy of stored scores       plain %.4f  centred(raw) %.4f' % (macro(ref_p.argmax(1)), macro(ref_c.argmax(1))))
rep = json.loads(Path('/kaggle/working/probe_native.json').read_text()); print(json.dumps(rep, indent=1))
print('seconds per passage per fold: %.3f  ->  4000 test passages x 5 folds ~ %.1f h' % (rep['fold_seconds'][0]['seconds'] / rep['passages'], 4000 * 5 * rep['fold_seconds'][0]['seconds'] / rep['passages'] / 3600))"""))
nb = new_notebook(cells=cells); nb.metadata = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}
nb.nbformat_minor = 4
nbformat.write(nb, '/home/rohan/kagglecomp/notebooks/native_probe.ipynb')
print('ok', len(cells))
