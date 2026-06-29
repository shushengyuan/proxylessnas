#!/usr/bin/env bash
set -euo pipefail

cd /home/intern/proxylessnas

GPU_IDS="${GPU_IDS:-1}"
LOG_ROOT="${LOG_ROOT:-/home/intern/proxylessnas/search/logs/proxyless_aligned_hna_darts_1500ep}"
DATA_ROOT="${DATA_ROOT:-/home/intern/proxylessnas/datasetyhy}"
GENE_PATH="${GENE_PATH:-/home/intern/proxylessnas/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-300}"
SEARCH_EPOCHS="${SEARCH_EPOCHS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LAUNCH_LOG="${LAUNCH_LOG:-${LOG_ROOT}/launch_logs/proxyless_aligned_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "${LAUNCH_LOG}")"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "[launch] $(date '+%F %T')"
echo "[launch] GPU_IDS=${GPU_IDS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
echo "[launch] LOG_ROOT=${LOG_ROOT}"
echo "[launch] BATCH_SIZE=${BATCH_SIZE} WARMUP_EPOCHS=${WARMUP_EPOCHS} SEARCH_EPOCHS=${SEARCH_EPOCHS}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

python search/SIRST_main_all.py \
  --path "${LOG_ROOT}" \
  --target_hardware gpu \
  --gpu "${GPU_IDS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --n_epochs "${SEARCH_EPOCHS}" \
  --retrain_epoch 1 \
  --model_name Super_all \
  --mode search \
  --dataset NUAA-SIRST \
  --root "${DATA_ROOT}" \
  --split_method 50_50 \
  --train_batch_size "${BATCH_SIZE}" \
  --test_batch_size "${BATCH_SIZE}" \
  --candidates_type Proxyless \
  --fast False \
  --arch_lr 0.005 \
  --arch_adam_beta1 0.0 \
  --grad_reg_loss_type 'mul#log' \
  --random_choose False \
  --gene "${GENE_PATH}"
