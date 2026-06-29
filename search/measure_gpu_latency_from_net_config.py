#!/usr/bin/env python3
"""
Measure GPU forward latency from net.config and save every sample to disk.

The default timing style follows search/run_manager.py's GPU branch:
  - batch size = 1
  - GPU input tensor
  - warmup + per-sample time.time() measurements
  - result unit = ms

Use --sync-cuda for a stricter CUDA timing measurement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from measure_cpu_time_from_net_config import (
    load_proxyless_builder,
    read_json,
    resolve_crop_size,
    resolve_gene,
    setup_paths,
)


def default_output_path(net_config_path: Path, samples: int, device: str, crop_size: int, sync_cuda: bool) -> Path:
    suffix = "sync" if sync_cuda else "legacy"
    safe_device = device.replace(":", "")
    if net_config_path.name == "net.config":
        return net_config_path.with_name(f"gpu_latency_{safe_device}_crop{crop_size}_{samples}samples_{suffix}.json")
    return net_config_path.with_name(
        f"{net_config_path.stem}.gpu_latency_{safe_device}_crop{crop_size}_{samples}samples_{suffix}.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure GPU forward latency from net.config and save every sample."
    )
    parser.add_argument("net_config", type=Path, help="Path to net.config")
    parser.add_argument("--gene", type=str, default=None, help="Path to phase1_gene.txt")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crop-size", type=int, default=None, help="Input H=W, default from run.config or 256")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--samples", type=int, default=200, help="Measured iterations")
    parser.add_argument("--sync-cuda", action="store_true", help="Synchronize CUDA before/after each timed forward")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable torch.backends.cudnn.benchmark")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, search_root = setup_paths()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    net_config_path = args.net_config.expanduser().resolve()
    if not net_config_path.is_file():
        raise FileNotFoundError(f"net.config not found: {net_config_path}")

    ProxylessNASNets = load_proxyless_builder(search_root)

    net_cfg = read_json(net_config_path)
    gene_path = resolve_gene(search_root, net_config_path, net_cfg, args.gene)
    crop_size = resolve_crop_size(net_config_path, args.crop_size)
    in_channels = int(net_cfg["blocks"][0].get("in_channels", 3))

    net = ProxylessNASNets.build_from_config(net_cfg, str(gene_path))
    net.to(device)
    net.eval()
    images = torch.zeros((1, in_channels, crop_size, crop_size), device=device)

    sample_times: list[float] = []
    with torch.no_grad():
        for idx in range(args.warmup + args.samples):
            if args.sync_cuda:
                torch.cuda.synchronize(device)
            start = time.time()
            net(images)
            if args.sync_cuda:
                torch.cuda.synchronize(device)
            used_ms = (time.time() - start) * 1e3
            if idx >= args.warmup:
                sample_times.append(used_ms)

    mean_ms = sum(sample_times) / len(sample_times)
    var_ms = sum((x - mean_ms) ** 2 for x in sample_times) / len(sample_times)
    std_ms = math.sqrt(var_ms)

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else default_output_path(net_config_path, args.samples, args.device, crop_size, args.sync_cuda)
    )
    report = {
        "net_config": str(net_config_path),
        "gene": str(gene_path),
        "crop_size": crop_size,
        "in_channels": in_channels,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device),
        "warmup": args.warmup,
        "samples": args.samples,
        "sync_cuda": args.sync_cuda,
        "cudnn_benchmark": args.cudnn_benchmark,
        "timing_source": "search/run_manager.py::RunManager.net_latency(l_type='gpu')"
        if not args.sync_cuda
        else "synchronized CUDA forward timing",
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "var_ms2": var_ms,
        "min_ms": min(sample_times),
        "max_ms": max(sample_times),
        "samples_ms": sample_times,
    }
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"net.config : {net_config_path}")
    print(f"gene       : {gene_path}")
    print(f"input      : batch=1 x {in_channels}x{crop_size}x{crop_size}, device={args.device}")
    print(f"warmup     : {args.warmup}")
    print(f"samples    : {args.samples}")
    print(f"sync_cuda  : {args.sync_cuda}")
    print(f"cudnn.benchmark: {args.cudnn_benchmark}")
    print(f"mean_ms    : {mean_ms:.6f}")
    print(f"std_ms     : {std_ms:.6f}")
    print(f"min_ms     : {min(sample_times):.6f}")
    print(f"max_ms     : {max(sample_times):.6f}")
    print(f"saved_json : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
