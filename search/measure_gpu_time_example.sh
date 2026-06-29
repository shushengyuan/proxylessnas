#!/usr/bin/env bash
# 一键示例：测量 GPU 前向时间（与 train_whole.sh 相同 conda / PYTHONPATH）
# Usage: bash measure_gpu_time_example.sh
# 可选：CUDA_VISIBLE_DEVICES=0 bash measure_gpu_time_example.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="${CONDA_BASE:-/home/intern/anaconda3}"
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
fi
# conda activate proxylessnas

cd "$SCRIPT_DIR"
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# 默认 net.config：与同级 parameters.txt 中 gene / iterations 一致，可省略 --gene 直接成功
NET_CONFIG="${NET_CONFIG:-$SCRIPT_DIR/logs/6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k/learned_net/net.config}"

python measure_gpu_time_from_net_config.py "$NET_CONFIG" --inference-repeated 400 \
  --gene ../search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt
