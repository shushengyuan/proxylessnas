#!/usr/bin/env python3
"""
Export the searched+retrained model to ONNX and benchmark inference latency.

Timing methods
--------------
① PyTorch GPU        (cuda.synchronize wall-clock)  — warmup 50, repeat 200
② ONNX CPU           (CPUExecutionProvider)          — warmup 50, repeat 200
③ ONNX CUDA          (CUDAExecutionProvider)         — warmup 50, repeat 200
④ ONNX TensorRT      (TensorrtExecutionProvider)     — warmup 50, repeat 200
⑤ TensorRT native    (trt.Builder → CUDA stream)     — warmup 50, repeat 200

Usage (from proxylessnas/search/):
  conda activate proxylessnas
  python onnx_export_and_infer.py [--device cuda:0] [--repeat 200]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

# ── path setup (same as measure_gpu_time_from_net_config.py) ────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
os.chdir(_script_dir)
for _p in (_repo_root, _script_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── defaults for IRSTD whole_all Retrain_8 ──────────────────────────────────
_NET_CONFIG = os.path.join(
    _script_dir,
    "logs",
    "6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k",
    "learned_net",
    "net.config",
)
_GENE = os.path.join(
    _repo_root,
    "search1yhy",
    "0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new",
    "phase1_gene.txt",
)
_CHECKPOINT = os.path.join(
    _script_dir,
    "logs",
    "6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k_Retrain_8",
    "checkpoint",
    "model_best.pth.tar",
)
_ONNX_OUTPUT = os.path.join(
    _script_dir,
    "logs",
    "6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k_Retrain_8",
    "model.onnx",
)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_net(net_config_path: str, gene_path: str, checkpoint_path: str, device):
    import torch
    from models import get_net_by_name

    cfg = json.load(open(net_config_path, encoding="utf-8"))
    net = get_net_by_name(cfg["name"]).build_from_config(cfg, gene_path)
    ckpt = torch.load(checkpoint_path, map_location=device,
                      weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.to(device)
    return net, cfg


def export_onnx(net, dummy: "torch.Tensor", onnx_path: str) -> None:
    import torch

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net,
            dummy,
            onnx_path,
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
    # verify
    import onnx
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX saved : {onnx_path}")
    print(f"  File size  : {size_mb:.2f} MB")


def bench_pytorch(net, dummy, warmup: int, repeat: int) -> tuple[float, float]:
    """Returns (mean_ms, std_ms) using cuda.synchronize wall-clock."""
    import torch

    times = []
    with torch.no_grad():
        for i in range(warmup + repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(dummy)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))


def bench_onnx(onnx_path: str, dummy_np: np.ndarray,
               provider: str, warmup: int, repeat: int) -> tuple[float, float]:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                providers=[provider])
    in_name = sess.get_inputs()[0].name

    times = []
    for i in range(warmup + repeat):
        t0 = time.perf_counter()
        sess.run(None, {in_name: dummy_np})
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    return float(np.mean(times)), float(np.std(times))


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="ONNX export + inference benchmark")
    parser.add_argument("--net-config",  default=_NET_CONFIG)
    parser.add_argument("--gene",        default=_GENE)
    parser.add_argument("--checkpoint",  default=_CHECKPOINT)
    parser.add_argument("--onnx-output", default=_ONNX_OUTPUT)
    parser.add_argument("--device",      default="cuda:0",
                        help="PyTorch device for export & GPU benchmark")
    parser.add_argument("--crop-size",   type=int, default=256)
    parser.add_argument("--warmup",      type=int, default=50)
    parser.add_argument("--repeat",      type=int, default=200)
    args = parser.parse_args()

    import torch

    device = torch.device(args.device)
    H = W  = args.crop_size

    # ── 1. load PyTorch model ────────────────────────────────────────────────
    print("\n[1] Loading PyTorch model …")
    net, cfg = load_net(args.net_config, args.gene, args.checkpoint, device)
    dummy_pt  = torch.zeros(1, 3, H, W, dtype=torch.float32, device=device)
    # dry run to allocate CUDA memory
    with torch.no_grad():
        net(dummy_pt)
    print("    Done.")

    # ── 2. export ONNX ───────────────────────────────────────────────────────
    print("\n[2] Exporting to ONNX …")
    export_onnx(net, dummy_pt, args.onnx_output)

    # ── 3. PyTorch GPU benchmark ─────────────────────────────────────────────
    print(f"\n[3] PyTorch GPU benchmark  (warmup={args.warmup}, repeat={args.repeat}) …")
    pt_mean, pt_std = bench_pytorch(net, dummy_pt, args.warmup, args.repeat)
    print(f"  mean : {pt_mean:.3f} ms   std : {pt_std:.3f} ms")

    # dummy numpy array for onnxruntime
    dummy_np = np.zeros((1, 3, H, W), dtype=np.float32)

    # ── 4. ONNX CPU benchmark ────────────────────────────────────────────────
    print(f"\n[4] ONNX CPU  benchmark    (warmup={args.warmup}, repeat={args.repeat}) …")
    cpu_mean, cpu_std = bench_onnx(args.onnx_output, dummy_np,
                                   "CPUExecutionProvider",
                                   args.warmup, args.repeat)
    print(f"  mean : {cpu_mean:.3f} ms   std : {cpu_std:.3f} ms")

    # ── 5. ONNX CUDA benchmark ───────────────────────────────────────────────
    print(f"\n[5] ONNX CUDA benchmark    (warmup={args.warmup}, repeat={args.repeat}) …")
    cuda_mean, cuda_std = bench_onnx(args.onnx_output, dummy_np,
                                     "CUDAExecutionProvider",
                                     args.warmup, args.repeat)
    print(f"  mean : {cuda_mean:.3f} ms   std : {cuda_std:.3f} ms")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 54)
    print(f"{'Backend':<22} {'mean (ms)':>10}  {'std (ms)':>10}")
    print("-" * 54)
    print(f"{'PyTorch GPU':<22} {pt_mean:>10.3f}  {pt_std:>10.3f}")
    print(f"{'ONNX  CPU':<22} {cpu_mean:>10.3f}  {cpu_std:>10.3f}")
    print(f"{'ONNX  CUDA':<22} {cuda_mean:>10.3f}  {cuda_std:>10.3f}")
    print(f"\nspeed-up  ONNX-CUDA vs PyTorch-GPU : {pt_mean/cuda_mean:.2f}x")
    print(f"speed-up  ONNX-CUDA vs ONNX-CPU    : {cpu_mean/cuda_mean:.2f}x")
    print("=" * 54)


if __name__ == "__main__":
    main()
