#!/usr/bin/env bash
set -euo pipefail

cd /home/intern/proxylessnas

PYTHON_BIN="${PYTHON_BIN:-/home/intern/anaconda3/envs/new_env/bin/python}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
LOG_ROOT="${LOG_ROOT:-/home/intern/proxylessnas/search/logs}"
DATA_ROOT="${DATA_ROOT:-/home/intern/proxylessnas/datasetyhy}"
GENE_PATH="${GENE_PATH:-/home/intern/proxylessnas/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25/phase1_gene_unskip.txt}"
RETRAIN_LOG_PATH="${RETRAIN_LOG_PATH:-0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_18_22_29}"
N_WORKER="${N_WORKER:-2}"
LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-${LOG_ROOT}/launch_logs}"
LAUNCH_LOG="${LAUNCH_LOG:-${LAUNCH_LOG_DIR}/mbconv_only_rerun_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "${LAUNCH_LOG_DIR}"
exec > >(tee -a "${LAUNCH_LOG}") 2>&1

echo "[launch] $(date '+%F %T')"
echo "[launch] python=${PYTHON_BIN}"
echo "[launch] gpu=${GPU_IDS}"
echo "[launch] log_root=${LOG_ROOT}"
echo "[launch] gene=${GENE_PATH}"
echo "[launch] n_worker=${N_WORKER}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_IDS}}"
if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
  export PYTORCH_CUDA_ALLOC_CONF
fi

"${PYTHON_BIN}" /home/intern/proxylessnas/search/SIRST_main_all.py \
  --path "${LOG_ROOT}" \
  --manual_seed 0 \
  --final_prior_pi -1.0 \
  --retrain_init_ckpt "" \
  --init_lr 0.01 \
  --lr_schedule_type cosine \
  --dataset NUAA-SIRST \
  --train_batch_size 16 \
  --test_batch_size 8 \
  --valid_size 1 \
  --opt_type Adagrad \
  --momentum 0.9 \
  --weight_decay 4e-05 \
  --label_smoothing 0.1 \
  --model_init he_fout \
  --validation_frequency 100 \
  --print_frequency 10 \
  --n_worker "${N_WORKER}" \
  --resize_scale 0.08 \
  --distort_color normal \
  --id_mode TXT \
  --root "${DATA_ROOT}" \
  --split_method 50_50 \
  --base_size 256 \
  --crop_size 256 \
  --suffix .png \
  --eval_batch_size 1 \
  --num_class 1 \
  --iterations 5 \
  --conv_type res_add_new_new \
  --channel_num two \
  --in_channel 3 \
  --arch_algo grad \
  --arch_init_type normal \
  --arch_init_ratio 0.001 \
  --arch_opt_type adam \
  --arch_lr 0.005 \
  --arch_adam_beta1 0.0 \
  --arch_adam_beta2 0.999 \
  --arch_adam_eps 1e-08 \
  --arch_weight_decay 0.0 \
  --target_hardware gpu \
  --fast False \
  --grad_update_arch_param_every 5 \
  --grad_update_steps 1 \
  --grad_binary_mode full_v2 \
  --grad_reg_loss_type 'mul#log' \
  --grad_reg_loss_lambda 0.1 \
  --grad_reg_loss_alpha 0.2 \
  --grad_reg_loss_beta 0.3 \
  --ref_value 7000000000.0 \
  --random_choose False \
  --warmup_epochs 300 \
  --n_epochs 1500 \
  --retrain_epoch 1500 \
  --retrain_init_lr 0.05 \
  --retrain_fixed_lr 0.01 \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --gpu "${GPU_IDS}" \
  --retrain_log_path "${RETRAIN_LOG_PATH}" \
  --add_decoder False \
  --add_encoder0 True \
  --mode all_search_train \
  --model_name Super_all \
  --candidates_type MBConvOnly \
  --gene "${GENE_PATH}" \
  --ROC_thr 10 \
  --postprocess none \
  --Inference_resize True \
  --Inference_repeated 100
