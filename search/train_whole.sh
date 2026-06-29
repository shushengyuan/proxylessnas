#!/usr/bin/env bash
# Super_all + whole_all search aligned to the 2023 NUAA-SIRST search regime.
# Override DATA_ROOT / GENE_PATH / LOG_PARENT / MODE / N_WORKER as needed.
# Usage: bash train_whole.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/intern/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
conda activate proxylessnas

cd "$SCRIPT_DIR"
# 使 `import search.*` 可解析（models 等包相对 proxylessnas 根目录）
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Reduces fragmentation OOMs on long runs (see PyTorch CUDA memory notes).
# export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- set these for your machine ---
export CUDA_VISIBLE_DEVICES=6,7
# 与 SIRST_main_all.py 一致：--root 为数据集父目录；--path 为日志根目录
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../datasetyhy}"
GENE_PATH="${GENE_PATH:-$SCRIPT_DIR/../search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
LOG_PARENT="${LOG_PARENT:-$SCRIPT_DIR/logs}"
MODE="${MODE:-all_search_train}"

# Keep worker count aligned to the 2023 reference search unless explicitly overridden.
N_WORKER="${N_WORKER:-4}"

python SIRST_main_all.py \
  --mode "$MODE" \
  --model_name Super_all \
  --candidates_type whole_all \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --dataset NUAA-SIRST \
  --split_method 50_50 \
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
  --grad_reg_loss_type mul#log \
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
