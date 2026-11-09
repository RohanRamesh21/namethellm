#!/usr/bin/env bash
# One-time setup on a new machine (e.g. a GPU cluster): virtualenv + packages + sanity checks.
#   bash scripts/setup_cluster.sh
# Environment variables: PYTHON (default python3, needs >= 3.10), TORCH_INDEX (e.g. https://download.pytorch.org/whl/cu124 to pick the CUDA build).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
pip install -U pip wheel
# torch first, so pip does not pull a CUDA build that does not match the cluster driver
if [ -n "${TORCH_INDEX:-}" ]; then pip install torch --index-url "$TORCH_INDEX"; else pip install torch; fi
pip install -r requirements.txt
echo "--- checks"
python - <<'PY'
import torch, transformers, peft, sklearn
print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| sklearn', sklearn.__version__)
n = torch.cuda.device_count(); print('GPUs visible:', n, [f'{torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory/2**30:.0f} GB' for i in range(n)])
if n == 0: print('WARNING: no GPU visible here (login node?). Run the GPU steps inside a job.')
PY
if [ -z "${KAGGLE_USERNAME:-}" ] && [ ! -f "$HOME/.kaggle/kaggle.json" ] && [ ! -f "${KAGGLE_CONFIG_DIR:-/nonexistent}/kaggle.json" ]; then
  echo "WARNING: no Kaggle credentials. Create an API token on kaggle.com (Settings > API), then either put it in ~/.kaggle/kaggle.json (chmod 600) or export KAGGLE_USERNAME and KAGGLE_KEY. NEVER commit it."
fi
echo "Next: bash scripts/fetch_assets.sh data small   (accept the competition rules on kaggle.com first)"
