#!/usr/bin/env bash
# Super_all + whole：仅做 phase2 retrain（不再跑搜索）。
# 依赖：对应搜索目录下已有 learned_net/net.config、learned_net/run.config、logs/arch.log（基因）。
# 用法：
#   RETRAIN_LOG_PATH=4,5_NUAA-SIRST_Super_all_whole_all_29_03_2026_15_51_29 bash retrain_whole.sh
# 或先改下面默认值再：bash retrain_whole.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- 按机器修改 ---
export CUDA_VISIBLE_DEVICES=4,5
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/../datasetyhy}"
# 与搜索阶段相同的 Meta gene 目录（用于 decode_gene / build_from_config 读基因）
GENE_PATH="${GENE_PATH:-$SCRIPT_DIR/../search1yhy/0_all_search_1_NUAA-SIRST_DNANet_Meta/phase1_gene.txt}"
LOG_PARENT="${LOG_PARENT:-$SCRIPT_DIR/logs}"
# 已完成搜索的日志文件夹名（位于 LOG_PARENT 下，不要带 _Retrain 后缀）
RETRAIN_LOG_PATH="${RETRAIN_LOG_PATH:-4,5_NUAA-SIRST_Super_all_whole_all_29_03_2026_15_51_29}"

python SIRST_main_all.py \
  --mode retrain \
  --model_name Super_all \
  --candidates_type whole_all \
  --gpu "${CUDA_VISIBLE_DEVICES}" \
  --path "$LOG_PARENT" \
  --root "$DATA_ROOT" \
  --gene "$GENE_PATH" \
  --retrain_log_path "$RETRAIN_LOG_PATH" \
  --dataset NUAA-SIRST \
  --split_method 50_50 \
  --id_mode TXT \
  --train_batch_size 16 \
  --test_batch_size 1 \
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
  --n_worker 4 \
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
  --retrain_epoch 1500 \
  --retrain_init_lr 0.05 \
  --retrain_fixed_lr 0.01 \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --ROC_thr 10 \
  --postprocess none
