#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEARCH_DIR="$REPO_ROOT/search"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/results/proxyless}"
mkdir -p "$OUT_DIR"

CONDA_BASE="${CONDA_BASE:-/home/intern/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-proxylessnas_irstd}"
fi

GPU="${GPU:-0}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasetyhy}"
GENE_PATH="${GENE_PATH:-$REPO_ROOT/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
LOG_PARENT="${LOG_PARENT:-$SEARCH_DIR/logs}"

WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-20}"

cd "$SEARCH_DIR"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python SIRST_main_all.py \
  --mode search \
  --gpu "$GPU" \
  --manual_seed "$SEED" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --dataset "${DATASET:-NUAA-SIRST}" \
  --split_method "${SPLIT_METHOD:-50_50}" \
  --model_name "${MODEL_NAME:-Super_all}" \
  --candidates_type "${CANDIDATES_TYPE:-whole_all}" \
  --iterations "${ITERATIONS:-5}" \
  --train_batch_size "${TRAIN_BATCH_SIZE:-8}" \
  --test_batch_size "${TEST_BATCH_SIZE:-1}" \
  --arch_algo grad \
  --target_hardware "${TARGET_HARDWARE:-gpu}" \
  --grad_reg_loss_type "${GRAD_REG_LOSS_TYPE:-add#linear}" \
  --grad_reg_loss_lambda "${GRAD_REG_LOSS_LAMBDA:-0.1}" \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --n_epochs "$SEARCH_EPOCHS"

LATEST_RUN="$(python - "$LOG_PARENT" "${DATASET:-NUAA-SIRST}" "${MODEL_NAME:-Super_all}" "${CANDIDATES_TYPE:-whole_all}" <<'PY'
import os
import sys
root, dataset, model_name, candidates_type = sys.argv[1:]
rows = []
if os.path.isdir(root):
    for name in os.listdir(root):
        if "_Retrain" in name:
            continue
        if f"_{dataset}_{model_name}_{candidates_type}_" not in name:
            continue
        full = os.path.join(root, name)
        if os.path.isdir(full):
            rows.append((os.path.getmtime(full), name))
rows.sort()
if not rows:
    raise SystemExit(1)
print(rows[-1][1])
PY
)"

python "$SCRIPT_DIR/extract_proxyless_artifacts.py" \
  --logs-root "$LOG_PARENT" \
  --run-name "$LATEST_RUN" \
  --output "$OUT_DIR/proxyless_${LATEST_RUN}.json"

echo "Proxyless search completed. Artifact: $OUT_DIR/proxyless_${LATEST_RUN}.json"
