# Name That LLM: handoff (last updated 2026-11-09, evening; sections 9-11 are the newest)

Read this first when resuming. Detailed notes also live in Claude's memory folder
`/home/rohan/.claude/projects/-home-rohan-kagglecomp/memory/` (index: MEMORY.md; the submission log is `competition-submission-rules.md`,
experiment results are in `idea-test-results.md`). This file is the self-contained summary.

## 1. The competition in one paragraph
Kaggle "Name That LLM": 18-way attribution of which LLM wrote a passage. Metric: balanced accuracy. Code competition: the notebook reads
`/kaggle/input/competitions/name-that-llm/test.csv` (hidden test ~4,000 passages, columns `id,text`, prompts not in train, per-model counts UNEQUAL;
locally `name-that-llm/test.csv` is a 3-row stub) and writes `submission.parquet` (columns `id`, `model`). Limits: GPU notebook at most 9 h, internet OFF,
5 submissions/day (resets at UTC midnight), 2 FINAL submissions to pick before the deadline **2027-01-13**. Official train: 2,699 rows = 150 prompts x 18 models
(exactly one passage per prompt/model; `claude-fable-5` has 149).

## 2. Where we stand
| ref | date (UTC) | what | public LB |
|---|---|---|---|
| 58135152 | 11-08 16:58 | v1 Gemma LoRA x2 + n-gram + Qwen-emb + BERT, prompt-centred | 0.540 |
| 58136557 | 11-08 17:50 | v2 same + Hungarian assignment in estimated groups | 0.517 |
| 58137820 | 11-08 18:38 | probe: Gemma scores alone | 0.514 |
| 58139526 | 11-08 19:48 | v3 stage-2 Gemma + n-gram(pool+official) + frozen Qwen/BERT | 0.551 |
| **58153691** | 11-09 06:04 | **v5 = v3's blend + public native expert (this is the BEST)** | **0.657** |
4 submissions were left on 2026-11-09. Remember to mark the 2 final submissions on the Kaggle site before 2027-01-13.
GPU quota: weekly 30 h, resets 2026-11-14. User reported **6 h 54 min left** on 2026-11-09 after the native work.
**Scoring reruns of a submission do NOT count against quota** (the 1.5 h rerun of v5 did not); only commit/dev runs do.

## 3. How the best submission (v5) works
Notebook `notebooks/submission_native.ipynb`, kernel `chiefkeef999/ntl-submission-native` version 2, T4 x2, internet off. Push folder `kaggle_push/submission_native/`.
Inputs it attaches (kernel_sources + dataset + model): outputs of `chiefkeef999/notebook6fafdc1c20` (Gemma stage-1 training run: base model copy, wheels, pool rows),
`chiefkeef999/ntl-stage-frozen-models` (Qwen3-Embedding-0.6B + BERT weights), `chiefkeef999/ntl-stage2-official` (stage-2 Gemma adapter), public dataset
`lightningv08/ntl-olmo2-native24-final-inference-assets` and Kaggle model `qwen-lm/qwen-3-embedding/transformers/8b/1`.
Pipeline (all in the notebook):
1. `ntl_pipeline.run_pipeline` (source `experiments/ntl_pipeline.py`): Gemma 3 4B LoRA scores x2 + n-gram SVC (pool 63k rows + official) + logistic regression on Qwen3-Embedding-0.6B
   and BERT features. Test passages are grouped into prompts by Ward clustering of Qwen last-layer features, K picked by silhouette; each group's mean is subtracted
   from n-gram and embedding scores ("centring", the single biggest trick). Saves `parent_scores.npz` (ids, blended score matrix S, estimated groups, labels).
2. `experiments/native_worker.py` (own port of CHEN Han's public inference): Qwen3-Embedding-8B in fp16 split over both T4s, fold LoRA adapters applied one at a time,
   pooled at 8 depths, plain + centred heads, 512-token covering windows. All 5 fold adapters scored on every test passage.
3. `experiments/native_fusion.py`: per fold calibrate with the author's stored scales, centre the second head inside estimated groups of 2-18, mix at the stored weight,
   average the 5 folds' log-probs. Final = 0.35 * z(parent S) + 0.65 * z(native), z = divide by std. (`NATIVE_WEIGHT = 0.65`.)
Timings: commit run on the 3-row stub ~15 min (parent) + 1.5 min (native); real-test scoring rerun ~1.5 h (free of quota).
Rebuild the notebook: `python3 experiments/build_notebooks/build_sub_native.py` (starts from notebooks/submission_stage2.ipynb, no cell ids because nbformat_minor=4),
copy to `kaggle_push/submission_native/`, then push (section 6).

## 4. What we learned (numbers are balanced accuracy)
Validated locally with prompt-grouped CV / held-out prompts unless stated:
- Prompt centring is the key trick; estimated groups work as well as true groups on official folds (native expert: 0.681 vs 0.682; `experiments/native_groups_local.py`).
  But on the real LB grouping gains were much smaller than simulated (cause unknown).
- Gemma 3 4B LoRA: stage 1 0.538 held-out; stage 2 (fine-tuned on official rows) 0.567 held-out but only +0.011 live (0.540 -> 0.551).
- Hungarian assignment inside groups helped in simulation, HURT live (0.517 vs 0.540). Keep off (`USE_ASSIGNMENT=False`).
- Qwen3-8B LoRA trained on official rows only: 0.471 held-out (worse than Gemma). The public native expert is strong because it was PRETRAINED on ~196k external rows first.
- Native expert (public, CHEN Han): official-row out-of-fold 0.652 plain, 0.688 with centred head. My port matches the author's stored scores (99.3% / 99.8% argmax agreement,
  correlation 0.99995; probe kernel `chiefkeef999/ntl-native-probe`). Blend on the 539 rows stage 2 held out: Gemma 0.567, native 0.693, mix (native share 0.6-0.8) 0.70-0.71. Live: 0.657.
- Second public expert WeMM4 (`ntl-native-wemm4-fivefold-0679`): OOF 0.643 fused; mixing with qwen8 gives about +0.01 (CI includes 0). NOT worth porting.
- Style embedders (Wegmann, LUAR-MUD, LUAR-CRUD, StyleDistance): +0.03 on the weak frozen-only blend (0.524 -> 0.55), but StyleDistance shrank to +0.006 once LoRA models were in. Not used.
  Results: `experiments/results_style/results.md`.
- Pre-fitted public "svc4" (4-view SVC, 231,960 training rows INCLUDING the official rows): author-reported clean score (trained on pool only, scored on official rows) **0.497** per passage
  versus ~0.42-0.44 for our own n-gram. It cannot be validated locally (official rows are in its training set; in-sample accuracy 0.9989). Only a leaderboard A/B can measure it, and the
  expected gain inside the blend (parent has 35% weight) is smaller than leaderboard noise (~0.007).

## 5. Ranked next steps
A. **Add the public Qwen3-14B / Qwen3-8B / Llama-3.1-8B "L24" LoRA experts** (best lead, bigger job). Public dataset `dhruvtibarewal/nlm-l24-adapters-v5` (2.59 GB): `q14v5`, `q8v5`, `ll8v5`,
   each a full adapter plus `_f0.._f4` per-fold adapters with `heads.pt` (~175 MB each for 14B). Gaussian prompt heads: `gsq120k-trained/gsq120k-inference/gaussian/q14v5|q8v5` in the lightningv08 dataset.
   Reported inner scores in the author's calibration files (`public_assets/native_wemm4/fold*/calibration.json`, entries `qwen_base8` 0.682, `qwen_base14` 0.691; I did NOT verify these are the same
   models as q8v5/q14v5). Inference code to read: `public_notebooks/ntl-680-lightning-v2.ipynb` cell 15 (`core_worker.py`), cell 12 (`core_readout.py`), cell 8 (`gauss.py`);
   `public_notebooks/0-673-i-m-better.ipynb` cell 12 (`infer.py`). Base models are Kaggle models (`qwen-lm/qwen-3/transformers/14b-base/1`, etc.). The per-fold adapters allow a CLEAN check:
   score each fold's held-out official rows with its own adapter and compare against our expectations, exactly as done for the native expert. Plan: port one expert (q8v5 first, cheapest),
   probe on one fold, then add to the blend like the native expert. Needs two T4s for 14B in fp16 or 4-bit.
B. **Swap/add the pre-fitted svc4** as a parent member. Ready: `experiments/svc4_readout.py` (+ `experiments/safe_joblib.py`), data in
   `public_assets/prefit/ntl673-response-core/ntl673-response-core/core/svc4/` (76 MB, already downloaded; the Kaggle notebook would attach the lightningv08 dataset instead).
   Suggested use: score test passages with `Svc4.decision`, `log_softmax(decision * 4)`, centre by estimated group, add to `parts` in `ntl_pipeline.recipe` (keep the old n-gram too). One slot to A/B.
C. Retune `NATIVE_WEIGHT` (0.6-0.8 was flat on held-out rows) or the estimated-group method (the public stack uses TF-IDF + average linkage at distance 0.90, `style_groups` in cell 10 of the 0.680 notebook).
Dropped: bigger Gemma pool run (parent only has 35% weight), WeMM4 port, style embedders, more official-only training, assignment.

## 6. How to operate (commands)
Always `export KAGGLE_CONFIG_DIR=/home/rohan/kagglecomp` (the `kaggle.json` there is the user's key: never print or copy it).
- Push: `cd kaggle_push/<name> && kaggle kernels push -p . --accelerator NvidiaTeslaT4` (NvidiaTeslaT4 = T4 x2). Status: `kaggle kernels status chiefkeef999/<kernel>`.
- Output (ALWAYS use `--file-pattern`, full outputs are ~5 GB): `kaggle kernels output chiefkeef999/<kernel> -p DEST --file-pattern '<regex>'`; log JSON is `DEST/<kernel>.log`.
- Submit: `kaggle competitions submit name-that-llm -k chiefkeef999/<kernel> -v <version> -f submission.parquet -m "<message>"`; list: `kaggle competitions submissions name-that-llm`.
- Public dataset files: `kaggle datasets files lightningv08/ntl-olmo2-native24-final-inference-assets --page-size 200 --format csv` (paginated; full listing cached in `experiments/cache/native_dataset_files.csv`);
  single file: `kaggle datasets download <owner>/<ds> -f <path/in/ds> -p DEST -q`. The dataset is ~100 GB: never download it whole.
- Commit run behaviour: it uses the 3-row stub `test.csv`, so group estimation is skipped there; the real test is only used by the scoring rerun after you submit.
- Wait for a long job without polling: a background `until ! kaggle ... status | grep -qE "RUNNING|QUEUED"; do sleep 60; done`.
- The user has permission-in-principle for me to submit ("i give you perms to submit whatever"); keep one spare slot when possible.

## 7. Machine limits and safety
- Laptop: RTX 5060 8 GB VRAM, ~4.6 GB free disk (98% full), ~1 MB/s connection, `/tmp` is tmpfs and is wiped on reboot. Derived arrays live in `experiments/cache/`. Downloads: use `curl -sL -C -` with retries for Hugging Face.
- Downloaded pickles/joblib files are untrusted: load them only through `experiments/safe_joblib.py` (allow-list; I first scanned zlib-compressed files with a scanner that saw nothing, so do NOT use plain
  pickletools on joblib files). The Kaggle notebooks run in a sandbox, which bounds the risk there.
- Public notebooks often hide behavior behind private datasets or "decoy" weights (see `public_notebooks/0-673-i-m-better.ipynb` section 1). Our v5 uses only public weights and states its sources in the notebook header.

## 8. File map
- `experiments/ntl_pipeline.py` parent pipeline (also saves parent_scores.npz); `score_gemma.py` Gemma scorer; `native_worker.py` / `native_fusion.py` native expert port;
  `svc4_readout.py` + `safe_joblib.py` pre-fitted SVC reader; `native_groups_local.py`, `native_blend_local.py` local studies of centring and blend weight (use `public_assets/native_q8` and
  `kaggle_results_stage2_pred/`); `style_embedders.py`, `ngram_official.py`, `native_small.py`, `sim_unequal.py`, `test_pipeline_local.py` earlier studies; `build_notebooks/` scripts that regenerate the notebooks.
- `notebooks/` all Kaggle notebooks (submission_native is the current best); `kaggle_push/` push folders with kernel-metadata.json.
- `public_notebooks/` downloaded public notebooks (`ntl-680-lightning-v2` = native-expert stack, `0-673-i-m-better` = areebbb NTL-22c, others). `public_assets/` small pieces of public datasets (native_q8, native_wemm4, prefit).
- Result folders: `kaggle_results_*` (pulled kernel outputs; `kaggle_results_subnative2` is v5's commit output), `experiments/results_style/`.

## 9. Update 2026-11-09 evening: the L24 expert (step A, started)
- Wrote `experiments/l24_worker.py` (own port of the public L24 inference: base truncated to the first 24 layers for Qwen3-8B-Base, fp32 low-rank LoRA path, 8 pooled depths + LayerNorm, plain and centred heads)
  and `experiments/l24_fusion.py`. Unit-tested on a tiny random Qwen3 against a merged-weights reference (logit diff 0.004, 100% argmax agreement).
- Probe `chiefkeef999/ntl-l24-probe` (24 min for 5 adapters x 2,699 rows, 4.7 min per adapter): the adapter dataset does not say which prompts each fold held out, but the held-out fold of each adapter is
  recoverable (each adapter is the worst, by loss, on exactly 30 prompts; held-out loss 1.09 vs 0.37 elsewhere). Their folds are NOT the native expert's folds. Assignment saved in `experiments/cache/q8v5_fold_of_row.npy`,
  out-of-fold scores in `experiments/cache/q8v5_oof.npz`.
- Out-of-fold balanced accuracy of q8v5: plain 0.615, centred head with true groups 0.667 (estimated groups 0.665). Native expert alone 0.681. **Native + q8v5 (centred) at 50/50: 0.704, +0.023 over native alone
  (95% interval +0.013 to +0.033)**; the two agree on 68.5% of rows. On the 539 stage-2 held-out rows: current recipe 0.703 -> 0.717-0.721 with q8v5 added (interval includes 0 at that size). Script: `experiments/l24_blend_local.py`.
- New notebook `notebooks/submission_l24.ipynb` (builder `experiments/build_notebooks/build_sub_l24.py`, kernel `chiefkeef999/ntl-submission-l24`): v5 + q8v5. Final blend 0.30 parent + 0.35 native + 0.35 L24, each z-scored.
  Datasets/models it needs: `dhruvtibarewal/nlm-l24-adapters-v5` and `qwen-lm/qwen-3/transformers/8b-base/1` on top of v5's inputs.
- 14B probe `chiefkeef999/ntl-l24-probe-q14` (q14v5, keep 27 layers, bs 8) launched the same way; if its out-of-fold result is as good, add it as a second L24 expert (the notebook's `experts` list is built for that).
- Not yet done: ll8v5 (Llama-3.1-8B; base is in the lightningv08 dataset folder `llama31-prefix24`), the Gaussian prompt heads (`gsq120k-trained/.../gaussian/*`), the final (all-rows) adapter `q8v5`.

## 10. Update 2026-11-09 afternoon: q14v5 and the three-expert submission
- **Kaggle allows only ONE GPU notebook at a time on this account** (user, 2026-11-09). Pushed kernels queue behind each other (my L24 commit run sat QUEUED ~1 h behind a 44-min probe). Push one at a time and check `kaggle kernels status` first.
- q14v5 (Qwen3-14B-Base, first 27 of 40 layers, `experiments/l24_worker.py --keep 27 --bs 8`): probe `chiefkeef999/ntl-l24-probe-q14`, 8.6 min per adapter per 2,699 rows (44 min for 5). Fold assignment recovered by loss as for q8v5
  (30 prompts each, loss gap 0.70), saved in `experiments/cache/q14v5_fold_of_row.npy`. Out of fold with estimated groups: centred head **0.680** (native 0.681, q8v5 0.665), plain 0.635.
  Native + q14v5 50/50 0.713 (+0.032). Native + q8v5 + q14v5 equal weights **0.722**, +0.019 over native + q8v5 (CI +0.011 to +0.028). Model agreement 68-74%. Script: `experiments/l24_blend3_local.py`.
- Notebooks (builder `experiments/build_notebooks/build_sub_l24.py`, `--q14` adds the 14B expert): `notebooks/submission_l24.ipynb` (parent .30 / native .35 / q8v5 .35; kernel `chiefkeef999/ntl-submission-l24`, commit clean, NOT submitted) and
  `notebooks/submission_l24_q14.ipynb` (parent .20 / native .2667 / q8v5 .2667 / q14v5 .2667; kernel `chiefkeef999/ntl-submission-l24-q14`). Both commit runs clean on the 3-row stub (~13.5 min parent + 1.3 min native + 0.9 min q8v5 + 1.3 min q14v5).
- **Submission 7: ref 58165598, kernel ntl-submission-l24-q14 v1, submitted 2026-11-09 13:23 UTC, result pending.** Expect the scoring rerun to take ~3 h (parent ~25 min, native ~60, q8v5 ~35, q14v5 ~65); it does not count against quota.
  If it errors in scoring, fall back to submitting `ntl-submission-l24` (q8v5 only). 3 slots were left after it.
- Still not done: ll8v5 (Llama-3.1-8B; base lives in the lightningv08 dataset folder `llama31-prefix24`), the Gaussian prompt heads, final all-rows adapters, `NATIVE`/`L24` weight retuning, committing the new files to git.

## 11. Moving to a GPU cluster (A100 / RTX 6000 Pro): read this on the cluster
**Setup (about 10 minutes plus downloads):**
```
git clone https://github.com/RohanRamesh21/namethellm.git && cd namethellm
bash scripts/setup_cluster.sh                 # venv + packages; set TORCH_INDEX=https://download.pytorch.org/whl/cuXXX to match the cluster's CUDA
# put your Kaggle API token in ~/.kaggle/kaggle.json (chmod 600) or export KAGGLE_USERNAME / KAGGLE_KEY. Never commit it.
# accept the competition rules once on kaggle.com, then:
. .venv/bin/activate
bash scripts/fetch_assets.sh data small       # competition files + native expert metadata/OOF scores (a few MB)
bash scripts/fetch_assets.sh native l24 hf    # adapters (3.5 GB) and base models (about 60 GB: check disk/quota first; the Hugging Face repos are public)
```
`scripts/fetch_assets.sh` modes: data, small, native, l24, svc4, hf, all. Paths are the same as on the laptop (`name-that-llm/`, `public_assets/`, `models/`), so the scripts in `experiments/` work unchanged. Not in the repo by design: `kaggle.json`, the competition data, other authors' files.
Scripts that read pulled Kaggle outputs expect them under `kaggle_results_*/` (for example `l24_blend_local.py` reads `kaggle_results_l24/q8v5_official.npz`); on the cluster write worker outputs to those paths or edit the paths.

**Hard constraint that does not change.** This is a code competition. Every submission is a Kaggle notebook (internet off, 9 h, 2 x T4 15 GB or P100) that reads its weights from Kaggle datasets or models. Anything trained on the cluster must be uploaded (`kaggle datasets create -p DIR`, then attached to the notebook), and its inference must finish on Kaggle's T4s inside the time limit alongside everything else (the current v7 budget is about 3 h). Cluster speed does not carry over, so budget each added model by its Kaggle cost: native 8B about 60 min for 4,000 passages x 5 folds, q8v5 about 35 min, q14v5 about 65 min.
The Kaggle weekly GPU quota (30 h) and the one-GPU-notebook-at-a-time rule apply to Kaggle runs only, not to the cluster.

**Verify the ports on the new hardware first (10-20 min on one big GPU):**
```
python - <<'PYEND'
import json, pandas as pd
d = pd.read_csv('name-that-llm/train.csv'); rows = json.load(open('public_assets/native_q8/fold0/split.json'))['validation_rows']
pd.DataFrame({'id': [str(i) for i in rows], 'text': d.text.fillna('').astype(str).iloc[rows].values}).to_csv('query_fold0.csv', index=False)
PYEND
python experiments/native_worker.py --base models/Qwen3-Embedding-8B --assets public_assets/native_q8 --query query_fold0.csv --out native_fold0.npz --work /tmp/native_work --folds 0
# expect >= 99% argmax agreement with the author's stored scores in public_assets/native_q8/oof_reference.npz at those rows (plain head about 0.626 balanced accuracy)
```
`native_worker.py` now runs on one GPU with at least 22 GB (it still splits over two GPUs as on Kaggle). `l24_worker.py` sizes its memory budget from the card (13 GiB on 15 GB cards as on Kaggle, 85% of the card on big ones). L24 check: score all official rows with the five q8v5 fold adapters and run `experiments/l24_blend_local.py`; expect plain about 0.615 and centred about 0.665 out of fold. The Hugging Face copies of Qwen3-8B-Base and Qwen3-14B-Base are assumed identical to the Kaggle copies the adapters were trained against (not verified); the check above is what confirms it.

**What the cluster is for (ranked; none of it started):**
1. **Train our own experts at the scale Kaggle's quota blocked.** The public experts are far stronger than anything we trained (our Gemma 0.567 held-out, the 8B trained on official rows only 0.471) because they were pretrained on 100k+ external passages first. Training data: the Open Pool `crtproduction/name-that-llm-open-pool` (CC BY-SA 4.0), plus `dhruvtibarewal/nlm-pool-v5` and `nlm-matched-clean` (download with `kaggle datasets download`, check sizes first). The public notebooks ship inference only, so a training recipe has to be rebuilt. What is known:
   - Native expert (`public_assets/native_q8/study.json`): Qwen3-Embedding-8B, LoRA rank 16 alpha 32 dropout 0.05 on q,k,v,o,gate,up,down; instruction prefix `Instruct: Identify the language model that generated the text based on its writing style.` then a newline and `Query: `; last-token pooling at depths (3,6,9,12,18,24,30,36), a LayerNorm each, depth 36 L2-normalised; plain and centred linear heads (centred coefficient 0.5 on official rows, 0 on external); 196,190 external rows x 2 epochs (6,132 updates, effective batch 64, adapter lr 2e-4 in that phase, head lr 5e-4), then 2 epochs on the 2,699 official rows (4 prompt groups per update, adapter lr 5e-5); label smoothing 0.05, weight decay 0.01, gradient clip 1.0, max length 512; five prompt folds. `experiments/native_small.py` is our 0.6B version of this idea (batches hold whole prompts so the centred head can be trained).
   - L24 experts (`public_assets/l24/*/adapter_config.json` and `heads.pt`): base truncated to the first 24 layers (8B) or 27 (14B), LoRA rank 16 alpha 32 dropout 0.05 on all seven projections, mean pooling at layers (2,4,6,8,12,16,20,24) with the first token masked, a LayerNorm each, plain and centred heads, max 384 tokens, trained in two stages (pool, then official folds). Learning rates and epochs are not published.
   A sensible first target: retrain the q8v5 recipe on the pool with prompt folds and compare its out-of-fold score with the public one (0.665 centred) using `experiments/l24_blend_local.py`. Only then scale up.
2. **Finish the cheap public-expert additions with clean checks:** ll8v5 (Llama-3.1-8B; its base weights are in the lightningv08 dataset folder `llama31-prefix24`), the Gaussian prompt heads (`gsq120k-trained/gsq120k-inference/gaussian/*`), and the final all-rows adapters. Score each out of fold exactly as q8v5 and q14v5 were (sections 9 and 10).
3. **Retune the blend weights** with all out-of-fold scores in one place (native, q8v5 and q14v5 exist for all 2,699 rows; the parent only for 539).
Current best submission and its pending result are in sections 2 and 10; run `kaggle competitions submissions name-that-llm` first.
