#!/usr/bin/env python3
"""
根据 learned_net/net.config 构建网络，按 run_manager.py 中两处 GPU 计时方法实测耗时。

===============================================================
两种计时方法（均来自 search/run_manager.py）：

① net_latency(l_type='gpu', fast='True')  —— 对应训练日志里 "gpu: X.Xms"
   位置：run_manager.py  RunManager.net_latency()
   逻辑：batch=1；net.eval()；torch.no_grad()；
         先循环 n_warmup=5 次（fast 模式），再循环 n_sample=10 次；
         每次单独用 time.time() 计时（单位 ms）；
         返回 sample 均值，单位 ms。

② inference_latency(args)  —— 对应 search_train.inference() 打印的 gpu_avg_time
   位置：run_manager.py  RunManager.inference_latency()
   逻辑：batch=1（来自 inference_loader_latency，DataLoader batch_size=1）；
         net.eval()；torch.no_grad()；
         直接循环 Inference_repeated=100 次，整块计时；
         返回 (end - start) / Inference_repeated，单位秒。
   注意：原实现无 warmup，无 cuda.synchronize()。
===============================================================

依赖：proxylessnas conda 环境 + PYTHONPATH 指向 proxylessnas 根目录（同 train_whole.sh）。

示例：
  bash measure_gpu_time_example.sh

手动：
  cd /path/to/proxylessnas/search
  export PYTHONPATH=/path/to/proxylessnas:$PYTHONPATH
  conda activate proxylessnas
  python measure_gpu_time_from_net_config.py --inference-repeated 100

若省略 --gene，会读 learned_net 同级 parameters.txt 的 gene:--（并校验 iterations 是否与 net.config 一致）。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time


def _setup_paths() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root  = os.path.dirname(script_dir)
    os.chdir(script_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _read_gene_and_iterations_from_parameters(log_root: str) -> tuple[str | None, int | None]:
    path = os.path.join(log_root, "parameters.txt")
    if not os.path.isfile(path):
        return None, None
    gene = None
    iters = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("gene:--"):
                gene = line.split("gene:--", 1)[1].strip()
            if line.startswith("iterations:--"):
                try:
                    iters = int(line.split("iterations:--", 1)[1].strip())
                except ValueError:
                    pass
    return gene, iters


def _read_crop_size(net_config_path: str) -> int:
    """从 learned_net/run.config 读取 crop_size，找不到返回 256。"""
    sibling = os.path.join(os.path.dirname(os.path.abspath(net_config_path)), "run.config")
    if os.path.isfile(sibling):
        rc = json.load(open(sibling, encoding="utf-8"))
        return int(rc.get("crop_size", 256))
    return 256


# ── 方法① ────────────────────────────────────────────────────────────────────
def net_latency_gpu(net, images, n_warmup: int = 5, n_sample: int = 10) -> tuple[float, list]:
    """
    对齐 RunManager.net_latency(l_type='gpu', fast='True')。
    - net.eval(); torch.no_grad()
    - 先 warmup n_warmup 次，再 sample n_sample 次，逐次用 time.time() 计时（ms）
    - 返回 (sample 均值 ms, sample 列表 ms)
    训练日志里 "gpu: X.Xms" 就是这里的均值。
    """
    import torch

    net.eval()
    sample_times: list[float] = []
    with torch.no_grad():
        for i in range(n_warmup + n_sample):
            start_time = time.time()
            net(images)
            used_time  = (time.time() - start_time) * 1e3   # → ms（与 run_manager.py 完全一致）
            if i >= n_warmup:
                sample_times.append(used_time)
    net.train()
    mean_ms = sum(sample_times) / n_sample
    return mean_ms, sample_times


# ── 方法② ────────────────────────────────────────────────────────────────────
def inference_latency_gpu(net, images, inference_repeated: int = 100) -> tuple[float, float, float, list]:
    """
    对齐 RunManager.inference_latency() 的 GPU 分支，同时额外统计逐次耗时。
    - net.eval(); torch.no_grad()
    - 无 warmup；整块循环 inference_repeated 次，一次性计时（与原版对齐）
    - 同时逐次记录耗时，用于计算均值和方差
    - 返回 (bulk_avg_s, per_mean_ms, per_std_ms, per_sample_list_ms)
    """
    import torch

    net.eval()
    sample_times: list[float] = []
    with torch.no_grad():
        bulk_start = time.time()
        for _ in range(inference_repeated):
            t0 = time.time()
            net(images)
            sample_times.append((time.time() - t0) * 1e3)
        bulk_avg_s = (time.time() - bulk_start) / inference_repeated

    per_mean = sum(sample_times) / inference_repeated
    per_var  = sum((x - per_mean) ** 2 for x in sample_times) / inference_repeated
    per_std  = math.sqrt(per_var)
    return bulk_avg_s, per_mean, per_std, sample_times


# ── 入口 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    _setup_paths()

    parser = argparse.ArgumentParser(
        description="Measure GPU forward time from net.config (aligned with run_manager.py)"
    )
    parser.add_argument(
        "net_config",
        nargs="?",
        default=None,
        help="Path to learned_net/net.config（默认使用仓库中 iterations/gene 一致的示例）",
    )
    parser.add_argument("--gene",  type=str, default=None, help="phase1_gene.txt（需与训练时一致）")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument(
        "--inference-repeated", type=int, default=100,
        help="方法②循环次数，对应 --Inference_repeated，默认 100（与 SIRST_main_all 一致）",
    )
    parser.add_argument("--crop-size", type=int, default=None, help="输入分辨率 H=W（默认从 run.config 读）")
    args = parser.parse_args()

    if args.net_config is None:
        args.net_config = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs/6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05/learned_net/net.config",
        )

    net_config_path = os.path.abspath(args.net_config)
    if not os.path.isfile(net_config_path):
        raise SystemExit(f"net.config not found: {net_config_path}")

    log_root     = os.path.dirname(os.path.dirname(net_config_path))
    net_cfg      = json.load(open(net_config_path, encoding="utf-8"))
    net_iters    = int(net_cfg.get("iterations", -1))

    # ── 定位 gene ──
    gene_path = args.gene
    param_gene, param_iters = _read_gene_and_iterations_from_parameters(log_root)
    if not gene_path:
        if param_iters is not None and param_iters != net_iters:
            raise SystemExit(
                "未指定 --gene：parameters.txt 中 iterations 与 net.config 不一致，无法自动推断。\n"
                f"  net.config iterations={net_iters}, parameters.txt iterations={param_iters}\n"
                "请通过 --gene 传入与该 net.config 对应的 phase1_gene.txt。"
            )
        gene_path = param_gene
    if not gene_path or not os.path.isfile(gene_path):
        raise SystemExit(
            "需要 phase1 gene 文件。请传入 --gene，或确保同级 parameters.txt 含 gene:-- 行且 iterations 一致。\n"
            f"  log_root={log_root}"
        )

    # ── 输入尺寸：两种方法 batch 均为 1（与 net_latency / inference_loader_latency 一致） ──
    in_ch = 3
    blocks = net_cfg.get("blocks") or []
    if blocks and isinstance(blocks[0], dict):
        in_ch = int(blocks[0].get("in_channels", 3))
    hw = args.crop_size if args.crop_size else _read_crop_size(net_config_path)
    batch_size = 1   # net_latency: batch=1；inference_loader_latency: DataLoader batch_size=1

    import torch
    from models import get_net_by_name
    from search.utils.utils import weights_init_xavier

    if not torch.cuda.is_available():
        raise SystemExit("未检测到 CUDA，无法测量 GPU 时间。")

    try:
        net = get_net_by_name(net_cfg["name"]).build_from_config(net_cfg, gene_path)
    except (TypeError, ValueError, RuntimeError, KeyError) as e:
        raise SystemExit(
            "build_from_config 失败（通常是 gene 与 net.config 不匹配）。\n"
            f"  错误: {e}"
        ) from e
    net.apply(weights_init_xavier)
    net.to(torch.device(args.device))

    images = torch.zeros(batch_size, in_ch, hw, hw,
                         device=torch.device(args.device), dtype=torch.float32)

    print("=" * 60)
    print("net.config :", net_config_path)
    print("gene       :", os.path.abspath(gene_path))
    print(f"input      : batch={batch_size} x {in_ch}x{hw}x{hw}, device={args.device}")
    print("=" * 60)

    try:
        # ── 方法① ──
        n_sample = 200
        mean_ms, samples = net_latency_gpu(images=images, net=net, n_warmup=5, n_sample=n_sample)
        var_ms  = sum((x - mean_ms) ** 2 for x in samples) / n_sample
        std_ms  = math.sqrt(var_ms)
        print("\n[方法①] net_latency(l_type='gpu', fast='True')  →  训练日志 'gpu: X.Xms' 的来源")
        print(f"  warmup=5, sample={n_sample}, 逐次计时")
        print(f"  sample mean : {mean_ms:.4f} ms")
        print(f"  sample std  : {std_ms:.4f} ms")
        print(f"  sample var  : {var_ms:.4f} ms²")
        print(f"  sample list : {[f'{v:.3f}' for v in samples]} ms")

        # ── 方法② ──
        bulk_avg_s, per_mean, per_std, per_samples = inference_latency_gpu(
            net=net, images=images, inference_repeated=args.inference_repeated
        )
        per_var = per_std ** 2
        print(f"\n[方法②] inference_latency()  →  search_train.inference() 打印的 gpu_avg_time")
        print(f"  Inference_repeated={args.inference_repeated}, 无 warmup")
        print(f"  bulk gpu_avg_time : {bulk_avg_s:.6f} s  ({bulk_avg_s * 1000:.4f} ms)  ← 与原版对齐")
        print(f"  per-sample mean   : {per_mean:.4f} ms")
        print(f"  per-sample std    : {per_std:.4f} ms")
        print(f"  per-sample var    : {per_var:.4f} ms²")

    except RuntimeError as e:
        raise SystemExit(
            "前向失败（通道/shape 不符时通常是 gene 与 net.config 不是同一实验导出的）。\n"
            f"  错误: {e}"
        ) from e

    print("=" * 60)


if __name__ == "__main__":
    main()
