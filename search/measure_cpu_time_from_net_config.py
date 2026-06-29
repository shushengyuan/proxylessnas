#!/usr/bin/env python3
"""
Measure CPU forward latency from net.config and save every sample to disk.

This follows the timing style used by search/run_manager.py:
  - batch size = 1
  - CPU input tensor
  - warmup + per-sample time.time() measurements
  - result unit = ms
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types


def setup_paths() -> tuple[Path, Path]:
    search_root = Path(__file__).resolve().parent
    repo_root = search_root.parent
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/proxylessnas_matplotlib")
    for path in (str(repo_root), str(search_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return repo_root, search_root


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_proxyless_builder(search_root: Path):
    # Avoid executing search/models/__init__.py, which imports run_manager and can
    # create a circular import when the script is used standalone.
    for name, path in [
        ("search.models", search_root / "models"),
        ("search.models.normal_nets", search_root / "models" / "normal_nets"),
        ("search.models.super_nets", search_root / "models" / "super_nets"),
    ]:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg

    proxyless_mod = load_module(
        "search.models.normal_nets.proxyless_nets",
        search_root / "models" / "normal_nets" / "proxyless_nets.py",
    )
    load_module(
        "search.models.super_nets.super_proxyless_SIRST",
        search_root / "models" / "super_nets" / "super_proxyless_SIRST.py",
    )
    return proxyless_mod.ProxylessNASNets


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_gene_from_parameters(log_root: Path) -> tuple[str | None, int | None]:
    path = log_root / "parameters.txt"
    if not path.is_file():
        return None, None
    gene = None
    iterations = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("gene:--"):
                gene = line.split("gene:--", 1)[1].strip()
            elif line.startswith("iterations:--"):
                try:
                    iterations = int(line.split("iterations:--", 1)[1].strip())
                except ValueError:
                    pass
    return gene, iterations


def resolve_gene(search_root: Path, net_config_path: Path, net_cfg: dict, gene_arg: str | None) -> Path:
    if gene_arg:
        gene_path = Path(gene_arg).expanduser().resolve()
        if not gene_path.is_file():
            raise FileNotFoundError(f"gene file not found: {gene_path}")
        return gene_path

    log_root = net_config_path.parent.parent
    param_gene, param_iters = read_gene_from_parameters(log_root)
    net_iters = int(net_cfg.get("iterations", -1))
    if param_gene and (param_iters is None or param_iters == net_iters):
        gene_path = Path(param_gene).expanduser().resolve()
        if gene_path.is_file():
            return gene_path

    meta_path = search_root / "Latency_measurement_meta.json"
    if meta_path.is_file():
        try:
            meta = read_json(meta_path)
            meta_gene = meta.get("build", {}).get("gene")
            if meta_gene:
                gene_path = Path(meta_gene).expanduser().resolve()
                if gene_path.is_file():
                    return gene_path
        except (json.JSONDecodeError, OSError):
            pass

    raise FileNotFoundError(
        "could not resolve phase1 gene file; pass --gene explicitly"
    )


def resolve_crop_size(net_config_path: Path, crop_size_arg: int | None) -> int:
    if crop_size_arg is not None:
        return int(crop_size_arg)
    run_config_path = net_config_path.parent / "run.config"
    if run_config_path.is_file():
        run_cfg = read_json(run_config_path)
        return int(run_cfg.get("crop_size", 256))
    return 256


def default_output_path(net_config_path: Path, samples: int) -> Path:
    if net_config_path.name == "net.config":
        return net_config_path.with_name(f"cpu_latency_{samples}samples.json")
    return net_config_path.with_name(f"{net_config_path.stem}.cpu_latency_{samples}samples.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CPU forward latency from net.config and save every sample."
    )
    parser.add_argument("net_config", type=Path, help="Path to net.config")
    parser.add_argument("--gene", type=str, default=None, help="Path to phase1_gene.txt")
    parser.add_argument("--crop-size", type=int, default=None, help="Input H=W, default from run.config or 256")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--samples", type=int, default=100, help="Measured iterations")
    parser.add_argument("--cpu-threads", type=int, default=1, help="torch.set_num_threads value")
    parser.add_argument("--output", type=Path, default=None, help="Where to save the JSON report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, search_root = setup_paths()

    net_config_path = args.net_config.expanduser().resolve()
    if not net_config_path.is_file():
        raise FileNotFoundError(f"net.config not found: {net_config_path}")

    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)

    import torch

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(args.cpu_threads, 4)))
    except RuntimeError:
        pass

    ProxylessNASNets = load_proxyless_builder(search_root)

    net_cfg = read_json(net_config_path)
    gene_path = resolve_gene(search_root, net_config_path, net_cfg, args.gene)
    crop_size = resolve_crop_size(net_config_path, args.crop_size)
    in_channels = int(net_cfg["blocks"][0].get("in_channels", 3))

    net = ProxylessNASNets.build_from_config(net_cfg, str(gene_path))
    net.eval()
    images = torch.zeros((1, in_channels, crop_size, crop_size), device="cpu")

    sample_times: list[float] = []
    with torch.no_grad():
        for idx in range(args.warmup + args.samples):
            start = time.time()
            net(images)
            used_ms = (time.time() - start) * 1e3
            if idx >= args.warmup:
                sample_times.append(used_ms)

    mean_ms = sum(sample_times) / len(sample_times)
    var_ms = sum((x - mean_ms) ** 2 for x in sample_times) / len(sample_times)
    std_ms = math.sqrt(var_ms)

    output_path = (args.output.expanduser().resolve() if args.output else default_output_path(net_config_path, args.samples))
    report = {
        "net_config": str(net_config_path),
        "gene": str(gene_path),
        "crop_size": crop_size,
        "in_channels": in_channels,
        "cpu_threads": args.cpu_threads,
        "warmup": args.warmup,
        "samples": args.samples,
        "timing_source": "search/run_manager.py::RunManager.net_latency(l_type='cpu')",
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
    print(f"input      : batch=1 x {in_channels}x{crop_size}x{crop_size}, cpu_threads={args.cpu_threads}")
    print(f"warmup     : {args.warmup}")
    print(f"samples    : {args.samples}")
    print(f"mean_ms    : {mean_ms:.6f}")
    print(f"std_ms     : {std_ms:.6f}")
    print(f"min_ms     : {min(sample_times):.6f}")
    print(f"max_ms     : {max(sample_times):.6f}")
    print(f"saved_json : {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
