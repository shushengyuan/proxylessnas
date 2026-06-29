#!/usr/bin/env bash
# Super_all + whole（IRSTD-SIRST）：仅 phase2 retrain，对应搜索目录：
#   logs/4,5_IRSTD-SIRST_Super_all_whole_all_30_03_2026_16_07_07
# 依赖：该目录下已有 learned_net/net.config、learned_net/run.config、logs/arch.log。
# 用法：
#   bash retrain_whole_irstd.sh
# 或覆盖日志名： RETRAIN_LOG_PATH=其他文件夹名 bash retrain_whole_irstd.sh
# 关闭 final conv 先验偏置： FINAL_PRIOR_PI=-1 bash retrain_whole_irstd.sh
# 换 conda 环境： CONDA_ENV=proxylessnas_irstd bash retrain_whole_irstd.sh（默认 internvl）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/intern/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
CONDA_ENV="${CONDA_ENV:-internvl}"
conda activate "$CONDA_ENV"

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- 按机器修改 ---
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../datasetyhy}"
# --gene 必须是 phase1_gene.txt 文件路径，不能是目录（否则会 IsADirectoryError）
GENE_PATH="${GENE_PATH:-$SCRIPT_DIR/../search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
LOG_PARENT="${LOG_PARENT:-$SCRIPT_DIR/logs}"
RETRAIN_LOG_PATH="${RETRAIN_LOG_PATH:-2,3_NUAA-SIRST_Super_all_whole_all_16_04_2026_21_03_02}"
# Mask R-CNN-style final-conv bias prior (same as BasicIRSTD ResUNet init); set FINAL_PRIOR_PI=-1 to disable.
FINAL_PRIOR_PI="${FINAL_PRIOR_PI:-0.0015}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
RETRAIN_FIXED_LR="${RETRAIN_FIXED_LR:-0.01}"
RETRAIN_INIT_LR="${RETRAIN_INIT_LR:-0.05}"
RETRAIN_EPOCH="${RETRAIN_EPOCH:-1500}"
RETRAIN_INIT_CKPT="${RETRAIN_INIT_CKPT:-}"

python SIRST_main_all.py \
  --mode retrain \
  --model_name Super_all \
  --candidates_type whole_all \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --retrain_log_path "$RETRAIN_LOG_PATH" \
  --dataset IRSTD-SIRST \
  --split_method 80_20 \
  --id_mode TXT \
  --train_batch_size 16 \
  --test_batch_size 16 \
  --valid_size 1 \
  --init_lr 0.01 \
  --lr_schedule_type fixed \
  --opt_type Adagrad \
  --momentum 0.9 \
  --weight_decay 4e-5 \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --model_init he_fout \
  --validation_frequency 1 \
  --print_frequency 10 \
  --n_worker 4 \
  --resize_scale 0.08 \
  --distort_color normal \
  --base_size 256 \
  --crop_size 256 \
  --suffix .png \
  --eval_batch_size 16 \
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
  --fast True \
  --grad_update_arch_param_every 5 \
  --grad_update_steps 1 \
  --grad_binary_mode full_v2 \
  --grad_reg_loss_type add#linear \
  --grad_reg_loss_lambda 0.1 \
  --grad_reg_loss_alpha 0.2 \
  --grad_reg_loss_beta 0.3 \
  --ref_value 7000000000 \
  --random_choose False \
  --retrain_epoch "${RETRAIN_EPOCH}" \
  --retrain_init_lr "${RETRAIN_INIT_LR}" \
  --retrain_fixed_lr "${RETRAIN_FIXED_LR}" \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --final_prior_pi "${FINAL_PRIOR_PI}" \
  --retrain_init_ckpt "${RETRAIN_INIT_CKPT}" \
  --ROC_thr 10 \
  --postprocess none
