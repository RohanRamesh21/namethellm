import copy, nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
nb = nbformat.read('/home/rohan/kagglecomp/notebooks/submission_stage2.ipynb', as_version=4)
cells = nb.cells

cells[0].source = """# Name That LLM: offline submission (stage-2 Gemma blend + public native expert)

Predicts the model behind each test passage and writes `submission.parquet`. Everything runs with **internet off**.

**Parent blend** (unchanged from the stage-2 submission, public score 0.551): the stage-2 Gemma 3 4B LoRA classifier (scores counted twice), an n-gram SVC trained on the pool rows and the official rows, and logistic regressions on Qwen3-Embedding and BERT features. Passages are grouped into prompts by clustering their text and each group's average is subtracted from the n-gram scores and the embedding features.

**New: the native expert.** CHEN Han's published Qwen3-Embedding-8B classifier (five prompt-fold rank-16 LoRA adapters, pooled at eight depths, a plain and a prompt-centred head; weights from the public dataset `lightningv08/ntl-olmo2-native24-final-inference-assets`, which re-hosts the author's `concyclics/ntl-native-q8-multidepth-fivefold` files unchanged). The weights were trained by the original author on about 196,000 external passages and then on the official rows. This notebook only runs inference: `native_worker.py` is my own port, which reproduced the author's stored out-of-fold scores on 540 held-out rows with 99.3% (plain head) and 99.8% (centred head) agreement. All five fold adapters are scored on every test passage and averaged. The centred head is centred inside the same estimated prompt groups the parent blend uses.

**Blend.** `0.35 x z(parent) + 0.65 x z(native)`, where `z` divides by the standard deviation. On the 539 official rows stage 2 held out (30 prompts, scores for the expert are its stored out-of-fold values): Gemma alone 0.567, expert alone 0.693, a 60-80% expert weight 0.70-0.71. The weight is set slightly lower than the best because the parent here is stronger than Gemma alone.

**Inputs to add**
1. The **Name That LLM** competition data.
2. Notebook outputs: `notebook6fafdc1c20` (training run), `ntl-stage-frozen-models`, `ntl-stage2-official`.
3. The dataset **lightningv08/ntl-olmo2-native24-final-inference-assets** and the model **qwen-lm/qwen-3-embedding** (transformers, 8b, version 1).

**How to run it**: GPU **T4 x2**, **Internet off**, Save & Run All. The commit run uses a 3-row stand-in `test.csv`, so group estimation is skipped there; the scoring run uses the real test (about 4,000 passages). Measured on 540 passages: 97 s per fold, so about an hour for five folds on the real test, plus the parent pipeline.

The notebook stops with an error rather than quietly submitting the parent blend alone (`REQUIRE_NATIVE`). The parent's own predictions are kept in `submission_parent.parquet`."""

# inputs cell: add native paths
c1 = cells[1].source
c1 = c1.replace("REQUIRE_GEMMA = True", "NATIVE_WEIGHT = 0.65    # share of the final z-scored blend given to the native expert (0.6-0.8 scored about the same on held-out rows)\nREQUIRE_NATIVE = True  # stop with an error rather than submit the parent blend alone\nREQUIRE_GEMMA = True", 1)
c1 = c1.replace("for name, value in dict(", """def first(paths, what):
    for p in paths:
        if os.path.exists(p): return p
    raise SystemExit(f'Could not find {what}. Tried {paths}. /kaggle/input has: {sorted(os.listdir("/kaggle/input"))}')
ASSETS = first(['/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl-native-q8-multidepth-fivefold-0679',
                '/kaggle/input/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets/ntl-native-q8-multidepth-fivefold-0679'], 'the native expert assets (add the dataset lightningv08/ntl-olmo2-native24-final-inference-assets)')
NATIVE_BASE = first(['/kaggle/input/models/qwen-lm/qwen-3-embedding/transformers/8b/1', '/kaggle/input/qwen-3-embedding/transformers/8b/1'], 'the Qwen3-Embedding-8B model (add qwen-lm/qwen-3-embedding, transformers, 8b)')
for name, value in dict(ASSETS=ASSETS, NATIVE_BASE=NATIVE_BASE, """, 1)
assert 'ASSETS=ASSETS' in c1 and 'NATIVE_WEIGHT' in c1
cells[1].source = c1

# pipeline cell: use the updated file
pipe = open('/home/rohan/kagglecomp/experiments/ntl_pipeline.py').read()
cells[3].source = '%%writefile ntl_pipeline.py\n' + pipe.rstrip('\n')

# new cells after the pipeline run (cell 7), before the result section (cells 8, 9)
worker = open('/home/rohan/kagglecomp/experiments/native_worker.py').read()
fusion = open('/home/rohan/kagglecomp/experiments/native_fusion.py').read()
new = [
 new_markdown_cell("## Native expert: score every test passage with the five fold adapters, then blend"),
 new_code_cell('%%writefile native_worker.py\n' + worker.rstrip('\n')),
 new_code_cell('%%writefile native_fusion.py\n' + fusion.rstrip('\n')),
 new_code_cell("""import shutil
from native_fusion import native_logp, blend_parent_native
shutil.copy('/kaggle/working/submission.parquet', '/kaggle/working/submission_parent.parquet')
parent = np.load('/kaggle/working/parent_scores.npz', allow_pickle=False)
ids, labels, S_parent, groups = parent['ids'].tolist(), parent['labels'].tolist(), parent['S'], parent['groups']
test = pd.read_csv(f'{COMP}/test.csv'); test['text'] = test.text.fillna('').astype(str)
assert len(test) == len(ids) and test.id.astype(str).tolist() == ids, 'test rows differ from the ones the parent pipeline scored'
pd.DataFrame({'id': ids, 'text': test.text.values}).to_csv('/kaggle/working/native_query.csv', index=False)
Path('/kaggle/temp/work').mkdir(parents=True, exist_ok=True)
t = time.time()
p = subprocess.Popen([sys.executable, '-u', 'native_worker.py', '--base', NATIVE_BASE, '--assets', ASSETS, '--query', '/kaggle/working/native_query.csv',
                      '--out', '/kaggle/working/native_scores.npz', '--work', '/kaggle/temp/work', '--folds', '0,1,2,3,4'],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1'))
tail = []
for line in p.stdout:
    print(line, end=''); tail = (tail + [line])[-40:]
if p.wait():
    if REQUIRE_NATIVE: raise SystemExit('The native expert failed:\\n' + ''.join(tail[-25:]))
    print('!!! native expert failed, keeping the parent blend')
else:
    print(f'native expert wall time {(time.time() - t) / 60:.1f} min')
    N = native_logp('/kaggle/working/native_scores.npz', ASSETS, groups, labels)
    final = blend_parent_native(S_parent, N, NATIVE_WEIGHT)
    pred = final.argmax(1)
    sub = pd.DataFrame({'id': ids, 'model': [labels[i] for i in pred]})
    assert sub.id.is_unique and sub.model.notna().all() and set(sub.model) <= set(labels), 'invalid submission'
    sub.to_parquet('/kaggle/working/submission.parquet', index=False)
    print('agreement  parent vs final %.1f%%   native vs final %.1f%%   parent vs native %.1f%%' % (
        100 * (S_parent.argmax(1) == pred).mean(), 100 * (N.argmax(1) == pred).mean(), 100 * (S_parent.argmax(1) == N.argmax(1)).mean()))
    print('prediction counts:', sub.model.value_counts().to_dict())
    info = json.loads(Path('/kaggle/working/submission_info.json').read_text())
    info['native'] = {'weight': NATIVE_WEIGHT, 'minutes': (time.time() - t) / 60, 'prediction_counts': sub.model.value_counts().to_dict()}
    Path('/kaggle/working/submission_info.json').write_text(json.dumps(info, indent=1))
    print(f'wrote /kaggle/working/submission.parquet ({len(sub)} rows)')"""),
]
nb.cells = cells[:8] + new + cells[8:]
for c in nb.cells: c.pop("id", None)
nb.nbformat_minor = 4
nbformat.validate(nb)
nbformat.write(nb, '/home/rohan/kagglecomp/notebooks/submission_native.ipynb')
print('cells', len(nb.cells))
