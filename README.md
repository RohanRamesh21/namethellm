# Name That LLM (Kaggle): work log and pipeline

Code for the Kaggle competition "Name That LLM" (18-way attribution of which LLM wrote a passage; metric: balanced accuracy; code competition,
offline notebook, hidden test of about 4,000 passages). Best public leaderboard score so far: **0.657**.

**Start with [HANDOFF.md](HANDOFF.md)**: current state, how the best submission works, what was tested (with numbers), ranked next steps,
Kaggle CLI commands and gotchas.

## Layout
- `experiments/` pipeline and study scripts. Main ones: `ntl_pipeline.py` (parent blend), `native_worker.py` and `native_fusion.py` (inference port of a public
  Qwen3-Embedding-8B expert), `score_gemma.py`, `svc4_readout.py` with `safe_joblib.py`, `build_notebooks/` (regenerates the Kaggle notebooks).
- `notebooks/` the Kaggle notebooks (`submission_native.ipynb` is the current best).
- `kaggle_push/` one folder per Kaggle kernel with its `kernel-metadata.json`, used by `kaggle kernels push`.

## Not in this repo (see `.gitignore`)
- `kaggle.json` (API key), the competition data and rules (`name-that-llm/`, `rules/`): get them from Kaggle.
- `public_notebooks/` and `public_assets/`: other authors' public notebooks and pieces of public Kaggle datasets that were downloaded for study.
  The dataset pieces are re-downloadable with `kaggle datasets download <owner>/<dataset> -f <path> -p <dir>` (examples in HANDOFF.md).
- `kaggle_results_*/` and `experiments/cache/`: kernel outputs and derived arrays; regenerate with `kaggle kernels output ... --file-pattern ...`.

## Credits and licensing
The native-expert weights and design are CHEN Han's (public Kaggle notebook `concyclics/0-679`, re-hosted in the dataset `lightningv08/ntl-olmo2-native24-final-inference-assets`);
`native_worker.py` and `svc4_readout.py` are my own inference implementations of what those public notebooks do. Other public work used or studied is credited in HANDOFF.md.
No license has been chosen for this repository; add one before making it public, and check the licenses of the third-party assets named above.
