"""Builds notebooks/submission_l24.ipynb from notebooks/submission_native.ipynb (the 0.657 notebook): adds the public Qwen3-8B-Base 'L24' expert (q8v5)."""
import argparse, nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
ap = argparse.ArgumentParser()
ap.add_argument('--q14', action='store_true', help='also add the Qwen3-14B-Base L24 expert (q14v5)')
ap.add_argument('--out', default='/home/rohan/kagglecomp/notebooks/submission_l24.ipynb')
args = ap.parse_args()
nb = nbformat.read('/home/rohan/kagglecomp/notebooks/submission_native.ipynb', as_version=4)
cells = nb.cells
assert ''.join(cells[11].source).startswith('import shutil') and ''.join(cells[12].source).startswith('## Result')

cells[0].source = """# Name That LLM: offline submission (parent blend + native expert + L24 expert)

Predicts the model behind each test passage and writes `submission.parquet`. Everything runs with **internet off**. Built on the 0.657 notebook (`ntl-submission-native` v2).

**Parent blend** (stage-2 Gemma 3 4B LoRA x2, n-gram SVC, Qwen3-Embedding and BERT logistic regressions, prompt-centred with groups estimated from the text; public score 0.551 alone).

**Native expert** (CHEN Han's published Qwen3-Embedding-8B classifier, five prompt-fold adapters; dataset `lightningv08/ntl-olmo2-native24-final-inference-assets`). Inference only, my own port in `native_worker.py`.

**New: L24 expert.** The public Qwen3-8B-Base classifier from `dhruvtibarewal/nlm-l24-adapters-v5` (`q8v5`): the base model cut to its first 24 layers, a rank-16 LoRA, eight pooled depths with a LayerNorm each, and plain and prompt-centred heads. Inference only, my own port in `l24_worker.py`; all five fold adapters are scored on every test passage and averaged, using the centred head inside estimated groups of 2-18 passages and the plain head elsewhere.

**Why this blend.** On the 2,699 official rows, scored out of fold (each row by an adapter that never saw its prompt): native expert 0.681, L24 expert (centred head, estimated groups) 0.665, and the two together at 50/50 0.704, a gain of +0.023 (95% interval +0.013 to +0.033) over the native expert alone; they agree on only 68.5% of passages. On the 539 rows stage 2 held out, adding the L24 expert to the previous recipe moved 0.703 to 0.717-0.721 (interval includes zero at that size). Weights: parent 0.30, native 0.35, L24 0.35, each divided by its standard deviation first.

**Inputs to add**: the competition data; notebook outputs `notebook6fafdc1c20`, `ntl-stage-frozen-models`, `ntl-stage2-official`; datasets `lightningv08/ntl-olmo2-native24-final-inference-assets` and `dhruvtibarewal/nlm-l24-adapters-v5`; models `qwen-lm/qwen-3-embedding` (transformers, 8b) and `qwen-lm/qwen-3` (transformers, 8b-base). GPU T4 x2, internet off.

**Running it.** The commit run uses a 3-row stand-in test, so group estimation is skipped there. Measured timings on 2,699 passages: native expert 97 s per fold per 540 passages; L24 expert 4.7 min per adapter, so about 7 min per adapter and 35 min in all for 4,000 passages. The notebook stops with an error rather than quietly submitting a weaker blend (`REQUIRE_NATIVE`, `REQUIRE_L24`). The two-way (parent + native) predictions are kept in `submission_two_way.parquet`."""
if args.q14:
    cells[0].source += """

**This version has two L24 experts.** It adds `q14v5`, the Qwen3-14B-Base classifier (first 27 of 40 layers; 8.6 min per adapter per 2,699 passages, so about 13 min per adapter and an hour in all for 4,000 passages; extra model input `qwen-lm/qwen-3` transformers 14b-base). Out of fold on the 2,699 official rows: q14v5 centred head 0.680 (the native expert gets 0.681); native + q14v5 at 50/50 0.713; native + q8v5 + q14v5 at equal weights 0.722, which is +0.019 (interval +0.011 to +0.028) over native + q8v5. Weights here: parent 0.20, native 0.267, each L24 expert 0.267."""

c1 = cells[1].source
W_LINE = "W_PARENT, W_NATIVE, W_L24 = 0.20, 0.2667, 0.5333   # final z-scored blend weights; W_L24 is shared equally by the L24 experts" if args.q14 else "W_PARENT, W_NATIVE, W_L24 = 0.30, 0.35, 0.35   # final z-scored blend weights"
c1 = c1.replace("REQUIRE_NATIVE = True", W_LINE + "\nREQUIRE_L24 = True     # stop with an error rather than submit without the L24 expert\nREQUIRE_NATIVE = True", 1)
c1 = c1.replace("for name, value in dict(ASSETS=ASSETS, NATIVE_BASE=NATIVE_BASE, ", """L24_ADAPTERS = first(['/kaggle/input/datasets/dhruvtibarewal/nlm-l24-adapters-v5', '/kaggle/input/nlm-l24-adapters-v5'], 'the L24 adapters (add the dataset dhruvtibarewal/nlm-l24-adapters-v5)')
Q8_BASE = first(['/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1', '/kaggle/input/qwen-3/transformers/8b-base/1'], 'the Qwen3-8B-Base model (add qwen-lm/qwen-3, transformers, 8b-base)')
""" + ("""Q14_BASE = first(['/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1', '/kaggle/input/qwen-3/transformers/14b-base/1'], 'the Qwen3-14B-Base model (add qwen-lm/qwen-3, transformers, 14b-base)')
""" if args.q14 else "") + """for name, value in dict(L24_ADAPTERS=L24_ADAPTERS, Q8_BASE=Q8_BASE, """ + ("Q14_BASE=Q14_BASE, " if args.q14 else "") + """ASSETS=ASSETS, NATIVE_BASE=NATIVE_BASE, """, 1)
assert 'Q8_BASE=Q8_BASE' in c1 and 'W_L24' in c1 and (not args.q14 or 'Q14_BASE=Q14_BASE' in c1)
cells[1].source = c1

worker = open('/home/rohan/kagglecomp/experiments/l24_worker.py').read(); fusion = open('/home/rohan/kagglecomp/experiments/l24_fusion.py').read()
new = [
 new_markdown_cell("## L24 expert: score every test passage with the five q8v5 fold adapters, then the final blend"),
 new_code_cell('%%writefile l24_worker.py\n' + worker.rstrip('\n')),
 new_code_cell('%%writefile l24_fusion.py\n' + fusion.rstrip('\n')),
 new_code_cell("""from l24_fusion import l24_logp
shutil.copy('/kaggle/working/submission.parquet', '/kaggle/working/submission_two_way.parquet')   # parent + native, written by the cell above
experts = [dict(name='q8v5', base=Q8_BASE, keep=24, bs=16, adapters=[f'{L24_ADAPTERS}/q8v5_f{k}' for k in range(5)])]
EXTRA_EXPERTS
E = []
for e in experts:
    t = time.time(); out = f'/kaggle/working/l24_{e["name"]}.npz'
    p = subprocess.Popen([sys.executable, '-u', 'l24_worker.py', '--base', e['base'], '--adapters', ','.join(e['adapters']), '--query', '/kaggle/working/native_query.csv',
                          '--out', out, '--keep', str(e['keep']), '--bs', str(e['bs'])], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         env=dict(os.environ, TRANSFORMERS_OFFLINE='1', HF_HUB_OFFLINE='1'))
    tail = []
    for line in p.stdout:
        print(line, end=''); tail = (tail + [line])[-40:]
    if p.wait():
        if REQUIRE_L24: raise SystemExit(f'The L24 expert {e["name"]} failed:\\n' + ''.join(tail[-25:]))
        print(f'!!! L24 expert {e["name"]} failed, leaving it out'); continue
    E.append(l24_logp(out, groups)); print(f'{e["name"]} wall time {(time.time() - t) / 60:.1f} min')
if E:
    z = lambda a: a / a.std()
    final = W_PARENT * z(S_parent) + W_NATIVE * z(N) + W_L24 * np.mean([z(x) for x in E], axis=0)
    pred = final.argmax(1)
    sub = pd.DataFrame({'id': ids, 'model': [labels[i] for i in pred]})
    assert sub.id.is_unique and sub.model.notna().all() and set(sub.model) <= set(labels), 'invalid submission'
    sub.to_parquet('/kaggle/working/submission.parquet', index=False)
    two = pd.read_parquet('/kaggle/working/submission_two_way.parquet')
    print('agreement  final vs two-way %.1f%%   final vs L24 alone %.1f%%   native vs L24 %.1f%%' % (100 * (two.model.values == sub.model.values).mean(),
          100 * (E[0].argmax(1) == pred).mean(), 100 * (N.argmax(1) == E[0].argmax(1)).mean()))
    print('prediction counts:', sub.model.value_counts().to_dict())
    info = json.loads(Path('/kaggle/working/submission_info.json').read_text())
    info['l24'] = {'weights': [W_PARENT, W_NATIVE, W_L24], 'experts': [e['name'] for e in experts], 'prediction_counts': sub.model.value_counts().to_dict()}
    Path('/kaggle/working/submission_info.json').write_text(json.dumps(info, indent=1))
    print(f'wrote /kaggle/working/submission.parquet ({len(sub)} rows)')"""),
]
EXTRA = "experts.append(dict(name='q14v5', base=Q14_BASE, keep=27, bs=8, adapters=[f'{L24_ADAPTERS}/q14v5_f{k}' for k in range(5)]))" if args.q14 else ""
for c in new:
    if c.cell_type == 'code': c.source = c.source.replace('EXTRA_EXPERTS', EXTRA).replace('\n\n', '\n') if 'EXTRA_EXPERTS' in c.source and not EXTRA else c.source.replace('EXTRA_EXPERTS', EXTRA)
nb.cells = cells[:12] + new + cells[12:]
for c in nb.cells: c.pop('id', None)
nb.nbformat_minor = 4
nbformat.validate(nb); nbformat.write(nb, args.out); print('cells', len(nb.cells))
