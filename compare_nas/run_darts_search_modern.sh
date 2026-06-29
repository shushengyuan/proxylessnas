#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DARTS_DIR="${DARTS_DIR:-$REPO_ROOT/third_party/pt.darts}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/darts}"
mkdir -p "$OUT_DIR"

if [ ! -d "$DARTS_DIR" ]; then
  echo "DARTS_DIR not found: $DARTS_DIR" >&2
  exit 1
fi

CONDA_BASE="${CONDA_BASE:-/home/intern/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-proxylessnas_irstd}"
fi

cd "$DARTS_DIR"
# Avoid old pinned deps in pt.darts requirements.txt.
python -m pip install -q "tensorboardX>=2.6" "graphviz>=0.20"

RUN_NAME="${RUN_NAME:-cifar10_proxyless_compare_seed42}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-50}"
GPUS="${GPUS:-0}"

python search.py \
  --name "$RUN_NAME" \
  --dataset "${DATASET:-cifar10}" \
  --gpus "$GPUS" \
  --seed "$SEED" \
  --epochs "$EPOCHS" \
  --batch_size "${BATCH_SIZE:-64}" \
  --workers "${WORKERS:-4}" \
  --print_freq "${PRINT_FREQ:-50}"

python "$SCRIPT_DIR/extract_darts_genotype.py" \
  --run-dir "$DARTS_DIR/searchs/$RUN_NAME" \
  --output "$OUT_DIR/darts_${RUN_NAME}.json"

echo "DARTS search completed. Artifact: $OUT_DIR/darts_${RUN_NAME}.json"
