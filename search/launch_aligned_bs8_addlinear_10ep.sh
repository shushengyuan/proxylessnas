#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-20260607_bs8_addlinear_10ep_nowarmup}"
DATA_ROOT="${DATA_ROOT:-/home/intern/proxylessnas/datasetyhy}"
GENE_PATH="${GENE_PATH:-/home/intern/proxylessnas/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt}"
CONDA_BASE="${CONDA_BASE:-/home/intern/anaconda3}"
CONDA_ENV="${CONDA_ENV:-new_env}"

if [ "$#" -ne 1 ]; then
  echo "usage: $0 {res_group_spa_mbconv|darts|proxyless}" >&2
  exit 2
fi

case "$1" in
  res_group_spa_mbconv)
    REPO_ROOT="/home/intern/proxylessnas"
    LOG_PARENT="/home/intern/proxylessnas/search/logs/aligned_${RUN_ID}/res_group_spa_mbconv"
    GPU_IDS="${GPU_IDS:-0,1}"
    TARGET_HARDWARE="gpu"
    CANDIDATES_TYPE="Res_Group_Spa_MBConv"
    GRAD_BINARY_MODE="full_v2"
    LR_SCHEDULE_TYPE="cosine"
    INFERENCE_REPEATED="100"
    ;;
  darts)
    REPO_ROOT="/home/intern/proxylessnas"
    LOG_PARENT="/home/intern/proxylessnas/search/logs/aligned_${RUN_ID}/darts"
    GPU_IDS="${GPU_IDS:-2,3}"
    TARGET_HARDWARE="gpu"
    CANDIDATES_TYPE="DARTS"
    GRAD_BINARY_MODE="darts"
    LR_SCHEDULE_TYPE="fixed"
    INFERENCE_REPEATED="100"
    ;;
  proxyless)
    REPO_ROOT="/home/intern/nas/proxylessnas-master-SIRST-new-final_share"
    LOG_PARENT="/home/intern/nas/proxylessnas-master-SIRST-new-final_share/search/logs/aligned_${RUN_ID}/proxyless"
    GPU_IDS="${GPU_IDS:-4,5}"
    TARGET_HARDWARE="flops"
    CANDIDATES_TYPE="Proxyless"
    GRAD_BINARY_MODE="full_v2"
    LR_SCHEDULE_TYPE="cosine"
    INFERENCE_REPEATED="1"
    ;;
  *)
    echo "unknown run: $1" >&2
    exit 2
    ;;
esac

mkdir -p "${LOG_PARENT}/launch_logs"
LAUNCH_LOG="${LOG_PARENT}/launch_logs/${1}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[launch] $(date '+%F %T')"
  echo "[launch] run=$1"
  echo "[launch] repo=${REPO_ROOT}"
  echo "[launch] log_parent=${LOG_PARENT}"
  echo "[launch] gpu=${GPU_IDS}"
  echo "[launch] data_root=${DATA_ROOT}"
  echo "[launch] gene=${GENE_PATH}"
  echo "[launch] n_epochs=10 warmup_epochs=0 train_batch_size=8 test_batch_size=8 grad_reg_loss_type=add#linear"
} | tee -a "${LAUNCH_LOG}"

exec > >(tee -a "${LAUNCH_LOG}") 2>&1

if [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

cd "${REPO_ROOT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python search/SIRST_main_all.py \
  --path "${LOG_PARENT}" \
  --manual_seed 0 \
  --init_lr 0.01 \
  --lr_schedule_type "${LR_SCHEDULE_TYPE}" \
  --dataset NUAA-SIRST \
  --root "${DATA_ROOT}" \
  --split_method 50_50 \
  --base_size 256 \
  --crop_size 256 \
  --suffix .png \
  --train_batch_size 8 \
  --test_batch_size 8 \
  --valid_size 1 \
  --eval_batch_size 1 \
  --n_worker 4 \
  --resize_scale 0.08 \
  --distort_color normal \
  --opt_type Adagrad \
  --momentum 0.9 \
  --weight_decay 4e-05 \
  --label_smoothing 0.1 \
  --model_init he_fout \
  --validation_frequency 1 \
  --print_frequency 10 \
  --num_class 1 \
  --arch_algo grad \
  --arch_init_type normal \
  --arch_init_ratio 0.001 \
  --arch_opt_type adam \
  --arch_lr 0.005 \
  --arch_adam_beta1 0.0 \
  --arch_adam_beta2 0.999 \
  --arch_adam_eps 1e-08 \
  --arch_weight_decay 0.0 \
  --target_hardware "${TARGET_HARDWARE}" \
  --fast False \
  --grad_update_arch_param_every 5 \
  --grad_update_steps 1 \
  --grad_binary_mode "${GRAD_BINARY_MODE}" \
  --grad_reg_loss_type 'add#linear' \
  --grad_reg_loss_lambda 0.1 \
  --grad_reg_loss_alpha 0.2 \
  --grad_reg_loss_beta 0.3 \
  --ref_value 7000000000.0 \
  --random_choose False \
  --warmup_epochs 0 \
  --n_epochs 10 \
  --retrain_epoch 1 \
  --retrain_init_lr 0.05 \
  --retrain_fixed_lr 0.01 \
  --retrain_lr_schedule_type fixed \
  --retrain_valid_size 1 \
  --retrain_latency gpu \
  --gpu "${GPU_IDS}" \
  --retrain_log_path 0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26 \
  --add_decoder False \
  --add_encoder0 True \
  --mode search \
  --model_name Super_all \
  --candidates_type "${CANDIDATES_TYPE}" \
  --gene "${GENE_PATH}" \
  --ROC_thr 10 \
  --postprocess none \
  --Inference_resize True \
  --Inference_repeated "${INFERENCE_REPEATED}" \
  2>&1 | tee -a "${LAUNCH_LOG}"
