#!/usr/bin/env bash
# Super_all + whole_all:
# phase1 用 NUAA-SIRST search，phase2 用 IRSTD-SIRST retrain（IRSTD1K 常用配置）。
# GPU latency 固定复用已有的 NUAA whole_all LUT。
# Usage: bash train_whole_nuaa_search_irstd_retrain.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/proxylessnas_matplotlib}"

# --- 可按机器覆盖 ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../datasetyhy}"
GENE_PATH="${GENE_PATH:-$SCRIPT_DIR/../search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
LOG_PARENT="${LOG_PARENT:-$SCRIPT_DIR/logs}"
DEFAULT_LATENCY_GPU_YAML="$SCRIPT_DIR/logs/0,1_NUAA-SIRST_Super_all_whole_all_27_04_2026_00_04_38/Latency_gpu.yaml"
LATENCY_GPU_YAML="${LATENCY_GPU_YAML:-$DEFAULT_LATENCY_GPU_YAML}"
USE_PREBUILT_LATENCY_GPU_YAML="${USE_PREBUILT_LATENCY_GPU_YAML:-True}"

if [[ "$USE_PREBUILT_LATENCY_GPU_YAML" == "True" || "$USE_PREBUILT_LATENCY_GPU_YAML" == "true" || "$USE_PREBUILT_LATENCY_GPU_YAML" == "1" ]]; then
  if [ ! -f "$LATENCY_GPU_YAML" ]; then
    echo "[error] LATENCY_GPU_YAML not found: $LATENCY_GPU_YAML" >&2
    exit 1
  fi
  export PROXYLESSNAS_LATENCY_GPU_YAML="$LATENCY_GPU_YAML"
  echo "[latency] using prebuilt GPU LUT: $PROXYLESSNAS_LATENCY_GPU_YAML"
fi

SEARCH_DATASET="${SEARCH_DATASET:-NUAA-SIRST}"
SEARCH_SPLIT_METHOD="${SEARCH_SPLIT_METHOD:-50_50}"
RETRAIN_DATASET="${RETRAIN_DATASET:-IRSTD-SIRST}"
RETRAIN_SPLIT_METHOD="${RETRAIN_SPLIT_METHOD:-80_20}"
N_WORKER="${N_WORKER:-4}"

echo "[1/2] Search operators on ${SEARCH_DATASET} ..."
python SIRST_main_all.py \
  --mode search \
  --model_name Super_all \
  --candidates_type whole_all \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --dataset "$SEARCH_DATASET" \
  --split_method "$SEARCH_SPLIT_METHOD" \
  --id_mode TXT \
  --train_batch_size 16 \
  --test_batch_size 16 \
  --valid_size 1 \
  --init_lr 0.01 \
  --lr_schedule_type cosine \
  --opt_type Adagrad \
  --momentum 0.9 \
  --weight_decay 4e-5 \
  --label_smoothing 0.1 \
  --model_init he_fout \
  --validation_frequency 1 \
  --print_frequency 10 \
  --n_worker "$N_WORKER" \
  --resize_scale 0.08 \
  --distort_color normal \
  --base_size 256 \
  --crop_size 256 \
  --suffix .png \
  --eval_batch_size 1 \
  --num_class 1 \
  --iterations 5 \
  --in_channel 3 \
  --arch_algo grad \
  --arch_init_type normal \
  --arch_init_ratio 0.001 \
  --arch_opt_type adam \
  --arch_lr 0.005 \
  --arch_adam_beta1 0 \
  --arch_adam_beta2 0.999 \
  --arch_adam_eps 1e-8 \
  --arch_weight_decay 0 \
  --target_hardware gpu \
  --fast False \
  --grad_update_arch_param_every 5 \
  --grad_update_steps 1 \
  --grad_binary_mode full_v2 \
  --grad_reg_loss_type "mul#log" \
  --grad_reg_loss_alpha 0.2 \
  --grad_reg_loss_beta 0.3 \
  --random_choose False \
  --warmup_epochs 300 \
  --n_epochs 1500 \
  --retrain_epoch 1500 \
  --retrain_init_lr 0.05 \
  --retrain_fixed_lr 0.01 \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --ROC_thr 10 \
  --postprocess none

# 自动定位本次/最近一次 NUAA 的 search 日志目录，给 retrain 用
RETRAIN_LOG_PATH="$(python - "$LOG_PARENT" "$SEARCH_DATASET" <<'PY'
import os
import sys

log_parent, dataset = sys.argv[1], sys.argv[2]
if not os.path.isdir(log_parent):
    raise SystemExit(f"LOG_PARENT not found: {log_parent}")

candidates = []
for name in os.listdir(log_parent):
    full_path = os.path.join(log_parent, name)
    if not os.path.isdir(full_path):
        continue
    if f"_{dataset}_Super_all_whole_all_" not in name:
        continue
    if "_Retrain" in name:
        continue
    candidates.append((os.path.getmtime(full_path), name))

if not candidates:
    raise SystemExit(f"No search log found under {log_parent} for dataset={dataset}")

candidates.sort()
print(candidates[-1][1])
PY
)"

echo "[2/2] Retrain on ${RETRAIN_DATASET} with search log: ${RETRAIN_LOG_PATH}"
python SIRST_main_all.py \
  --mode retrain \
  --model_name Super_all \
  --candidates_type whole_all \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --retrain_log_path "$RETRAIN_LOG_PATH" \
  --dataset "$RETRAIN_DATASET" \
  --split_method "$RETRAIN_SPLIT_METHOD" \
  --id_mode TXT \
  --train_batch_size 16 \
  --test_batch_size 16 \
  --valid_size 1 \
  --init_lr 0.01 \
  --lr_schedule_type fixed \
  --opt_type Adagrad \
  --momentum 0.9 \
  --weight_decay 4e-5 \
  --label_smoothing 0.1 \
  --model_init he_fout \
  --validation_frequency 1 \
  --print_frequency 10 \
  --n_worker "$N_WORKER" \
  --resize_scale 0.08 \
  --distort_color normal \
  --base_size 256 \
  --crop_size 256 \
  --suffix .png \
  --eval_batch_size 1 \
  --num_class 1 \
  --iterations 5 \
  --in_channel 3 \
  --arch_algo grad \
  --arch_init_type normal \
  --arch_init_ratio 0.001 \
  --arch_opt_type adam \
  --arch_lr 0.005 \
  --arch_adam_beta1 0 \
  --arch_adam_beta2 0.999 \
  --arch_adam_eps 1e-8 \
  --arch_weight_decay 0 \
  --target_hardware gpu \
  --fast False \
  --grad_update_arch_param_every 5 \
  --grad_update_steps 1 \
  --grad_binary_mode full_v2 \
  --grad_reg_loss_type "mul#log" \
  --grad_reg_loss_alpha 0.2 \
  --grad_reg_loss_beta 0.3 \
  --random_choose False \
  --retrain_epoch 1500 \
  --retrain_init_lr 0.05 \
  --retrain_fixed_lr 0.01 \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --ROC_thr 10 \
  --postprocess none
