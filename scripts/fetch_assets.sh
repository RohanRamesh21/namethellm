#!/usr/bin/env bash
# Download what the repo needs but does not contain. Run from anywhere; everything lands under the repo root. Safe to re-run (existing files are skipped).
#   bash scripts/fetch_assets.sh MODE [MODE...]
# Modes (sizes approximate):
#   data    competition files -> name-that-llm/            (needs the competition rules accepted on kaggle.com; test.csv is a 3-row stub)
#   small   native expert metadata, calibration, OOF scores and tokenizer files -> public_assets/native_q8/   (~5 MB)
#   native  native expert fold adapters (5 x 180 MB) -> public_assets/native_q8/fold*/trainables.safetensors
#   l24     L24 expert adapters q8v5/q14v5/ll8v5 (2.6 GB) -> public_assets/l24/
#   svc4    pre-fitted 4-view SVC (76 MB) -> public_assets/prefit/...
#   hf      base models from Hugging Face -> models/   (Qwen3-Embedding-8B 16 GB, Qwen3-8B-Base 16 GB, Qwen3-14B-Base 28 GB)
#   all     everything above
# Needs the kaggle CLI (credentials: see setup_cluster.sh) and, for `hf`, the `hf` command from huggingface_hub.
set -euo pipefail
cd "$(dirname "$0")/.."
DS=lightningv08/ntl-olmo2-native24-final-inference-assets
NATIVE=ntl-native-q8-multidepth-fivefold-0679
fetch() {  # fetch DATASET PATH_IN_DATASET DEST_DIR
  local out="$3/$(basename "$2")"
  if [ -s "$out" ]; then return; fi
  mkdir -p "$3"; kaggle datasets download "$1" -f "$2" -p "$3" -q
  [ -s "$out" ] || { echo "download failed: $2" >&2; exit 1; }
}
do_data() {
  if [ ! -s name-that-llm/train.csv ]; then
    mkdir -p name-that-llm && kaggle competitions download -c name-that-llm -p name-that-llm && (cd name-that-llm && unzip -oq name-that-llm.zip && rm -f name-that-llm.zip)
  fi
  ls name-that-llm
}
do_small() {
  local d=public_assets/native_q8
  for f in manifest.json study.json ORIGIN.json oof_reference.npz; do fetch $DS $NATIVE/$f $d; done
  for k in 0 1 2 3 4; do fetch $DS $NATIVE/calibration/fold$k.json $d/calibration; fetch $DS $NATIVE/fold$k/split.json $d/fold$k; done
  for f in config.json tokenizer.json tokenizer_config.json merges.txt vocab.json model.safetensors.index.json; do fetch $DS $NATIVE/base/$f $d/base; done
}
do_native() { for k in 0 1 2 3 4; do fetch $DS $NATIVE/fold$k/trainables.safetensors public_assets/native_q8/fold$k; done; }
do_l24() {
  if [ ! -s public_assets/l24/q14v5_f4/adapter_model.safetensors ]; then
    mkdir -p public_assets/l24 && kaggle datasets download dhruvtibarewal/nlm-l24-adapters-v5 -p public_assets/l24 --unzip
  fi
  ls public_assets/l24 | head -30
}
do_svc4() {
  local b=ntl673-response-core/ntl673-response-core/core/svc4 d=public_assets/prefit/ntl673-response-core/ntl673-response-core/core/svc4
  for f in metadata.json model.npz vectorizer-0.joblib vectorizer-1.joblib vectorizer-2.joblib vectorizer-3.joblib; do fetch $DS $b/$f $d; done
}
do_hf() {
  # The native expert was trained on this exact revision (manifest.json). The L24 experts were trained on Qwen3-8B-Base / Qwen3-14B-Base; the revision is not recorded,
  # so verify the port on your hardware (probe: score the official rows and compare against the out-of-fold numbers in HANDOFF.md) before trusting these copies.
  hf download Qwen/Qwen3-Embedding-8B --revision 1d8ad4ca9b3dd8059ad90a75d4983776a23d44af --local-dir models/Qwen3-Embedding-8B --include "*.safetensors" "model.safetensors.index.json"
  hf download Qwen/Qwen3-8B-Base --local-dir models/Qwen3-8B-Base
  hf download Qwen/Qwen3-14B-Base --local-dir models/Qwen3-14B-Base
}
[ $# -ge 1 ] || { sed -n '2,15p' "$0"; exit 1; }
for m in "$@"; do
  case "$m" in
    data) do_data;; small) do_small;; native) do_native;; l24) do_l24;; svc4) do_svc4;; hf) do_hf;;
    all) do_data; do_small; do_native; do_l24; do_svc4; do_hf;;
    *) echo "unknown mode $m" >&2; exit 1;;
  esac
done
echo "done: $*"
