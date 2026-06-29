#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <candidates_type> <gpu_id> [log_suffix]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CANDIDATES_TYPE="$1"
GPU_ID="$2"
LOG_SUFFIX="${3:-400epoch_curve}"

PYTHON="${PYTHON:-/home/intern/anaconda3/envs/new_env/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasetyhy}"
GENE_PATH="${GENE_PATH:-$REPO_ROOT/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
LOG_PARENT="${LOG_PARENT:-$SCRIPT_DIR/logs}"
LATENCY_GPU_YAML="${LATENCY_GPU_YAML:-$SCRIPT_DIR/logs/0,1_NUAA-SIRST_Super_all_whole_all_27_04_2026_00_04_38/Latency_gpu.yaml}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-16}"
N_WORKER="${N_WORKER:-2}"
N_EPOCHS="${N_EPOCHS:-400}"
PRINT_FREQUENCY="${PRINT_FREQUENCY:-10}"
VALIDATION_FREQUENCY="${VALIDATION_FREQUENCY:-1}"
GRAD_BINARY_MODE="${GRAD_BINARY_MODE:-full_v2}"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/proxylessnas_matplotlib}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PROXYLESSNAS_LATENCY_GPU_YAML="$LATENCY_GPU_YAML"

cd "$REPO_ROOT"

echo "[run] candidates_type=${CANDIDATES_TYPE} gpu=${GPU_ID} suffix=${LOG_SUFFIX}"
echo "[run] python=${PYTHON}"
echo "[run] latency LUT=${PROXYLESSNAS_LATENCY_GPU_YAML}"

"$PYTHON" search/SIRST_main_all.py \
  --mode search \
  --model_name Super_all \
  --candidates_type "$CANDIDATES_TYPE" \
  --gpu "$GPU_ID" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --dataset NUAA-SIRST \
  --split_method 50_50 \
  --id_mode TXT \
  --train_batch_size "$TRAIN_BATCH_SIZE" \
  --test_batch_size "$TEST_BATCH_SIZE" \
  --valid_size 1 \
  --init_lr 0.01 \
  --lr_schedule_type cosine \
  --opt_type Adagrad \
  --momentum 0.9 \
  --weight_decay 4e-5 \
  --label_smoothing 0.1 \
  --model_init he_fout \
  --validation_frequency "$VALIDATION_FREQUENCY" \
  --print_frequency "$PRINT_FREQUENCY" \
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
  --grad_binary_mode "$GRAD_BINARY_MODE" \
  --grad_reg_loss_type mul#log \
  --grad_reg_loss_alpha 0.2 \
  --grad_reg_loss_beta 0.3 \
  --random_choose False \
  --warmup_epochs 0 \
  --n_epochs "$N_EPOCHS" \
  --retrain_epoch 1 \
  --retrain_init_lr 0.05 \
  --retrain_fixed_lr 0.01 \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --ROC_thr 10 \
  --postprocess none
