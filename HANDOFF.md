# Name That LLM: handoff (last updated 2026-11-09, ~15:00 local)

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
