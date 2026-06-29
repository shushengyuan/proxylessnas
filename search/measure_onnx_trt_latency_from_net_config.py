#!/usr/bin/env python3
"""
Export a ProxylessNAS net.config to ONNX and benchmark ONNXRuntime backends.

Backends:
  - ONNXRuntime CPUExecutionProvider
  - ONNXRuntime CUDAExecutionProvider
  - ONNXRuntime TensorrtExecutionProvider, FP32
  - ONNXRuntime TensorrtExecutionProvider, FP16

The benchmark uses fixed input shape batch=1 x C x crop_size x crop_size and
saves every measured sample to JSON.
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import math
import os
from pathlib import Path
import statistics
import time
import warnings

import numpy as np

from measure_cpu_time_from_net_config import (
    load_proxyless_builder,
    read_json,
    resolve_crop_size,
    resolve_gene,
    setup_paths,
)


def preload_tensorrt_libs() -> list[str]:
    loaded: list[str] = []
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        return loaded

    import tensorrt as trt

    candidates = []
    site_dir = Path(trt.__file__).resolve().parents[1]
    candidates.append(site_dir / "tensorrt_libs")
    candidates.append(site_dir / "tensorrt_cu12_libs")
    candidates.extend(Path(p) for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p)

    for lib_dir in candidates:
        if not lib_dir.is_dir():
            continue
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + ld_path if ld_path else "")
        for pattern in ("libnvinfer.so*", "libnvinfer_plugin.so*", "libnvonnxparser.so*"):
            for so_path in sorted(glob.glob(str(lib_dir / pattern))):
                try:
                    ctypes.CDLL(so_path)
                    loaded.append(so_path)
                    break
                except OSError:
                    pass
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure ONNXRuntime ONNX/TRT latency from net.config.")
    parser.add_argument("net_config", type=Path)
    parser.add_argument("--gene", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--onnx-path", type=Path, default=None)
    parser.add_argument("--no-cpu", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--no-trt", action="store_true")
    parser.add_argument("--trt-cache-dir", type=Path, default=None)
    return parser.parse_args()


def load_checkpoint_if_available(net, checkpoint_path: Path | None, device: str) -> dict:
    if checkpoint_path is None:
        return {"loaded": False, "path": None, "reason": "not provided"}
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        return {"loaded": False, "path": str(checkpoint_path), "reason": "file not found"}

    import torch

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    try:
        net.load_state_dict(state)
        return {"loaded": True, "path": str(checkpoint_path), "strict": True}
    except RuntimeError as first_error:
        if isinstance(state, dict):
            stripped = {
                key[len("module.") :] if key.startswith("module.") else key: value
                for key, value in state.items()
            }
            try:
                net.load_state_dict(stripped)
                return {
                    "loaded": True,
                    "path": str(checkpoint_path),
                    "strict": True,
                    "stripped_module_prefix": True,
                }
            except RuntimeError:
                pass
        return {
            "loaded": False,
            "path": str(checkpoint_path),
            "reason": str(first_error).splitlines()[:8],
        }


def export_onnx(net, images, output_path: Path) -> None:
    import onnx
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net,
            images,
            str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            do_constant_folding=True,
        )
    onnx.checker.check_model(onnx.load(str(output_path)))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize(samples: list[float]) -> dict:
    mean_ms = float(sum(samples) / len(samples))
    var_ms2 = float(sum((x - mean_ms) ** 2 for x in samples) / len(samples))
    return {
        "ok": True,
        "mean_ms": mean_ms,
        "std_ms": math.sqrt(var_ms2),
        "var_ms2": var_ms2,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "p90_ms": percentile(samples, 90),
        "p95_ms": percentile(samples, 95),
        "samples_ms": samples,
    }


def make_session(onnx_path: Path, provider: str, provider_options: dict):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 2
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=[(provider, provider_options)],
    )


def benchmark_provider(
    onnx_path: Path,
    images_np: np.ndarray,
    provider: str,
    provider_options: dict,
    warmup: int,
    samples: int,
) -> dict:
    import onnxruntime as ort

    if provider not in ort.get_available_providers():
        return {"ok": False, "error": f"{provider} is not available"}

    try:
        session = make_session(onnx_path, provider, provider_options)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}

    actual_providers = session.get_providers()
    if provider not in actual_providers:
        return {"ok": False, "error": f"session providers={actual_providers}"}

    input_name = session.get_inputs()[0].name
    times: list[float] = []
    try:
        for idx in range(warmup + samples):
            start = time.perf_counter()
            session.run(None, {input_name: images_np})
            used_ms = (time.perf_counter() - start) * 1e3
            if idx >= warmup:
                times.append(used_ms)
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "actual_providers": actual_providers}

    result = summarize(times)
    result.update({"provider": provider, "provider_options": provider_options, "actual_providers": actual_providers})
    return result


def main() -> None:
    args = parse_args()
    _, search_root = setup_paths()
    loaded_trt_libs = preload_tensorrt_libs()

    import onnxruntime as ort
    import torch

    net_config_path = args.net_config.expanduser().resolve()
    if not net_config_path.is_file():
        raise FileNotFoundError(f"net.config not found: {net_config_path}")

    if args.device.startswith("cuda"):
        device_id = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        torch.cuda.set_device(device_id)
    else:
        device_id = 0

    ProxylessNASNets = load_proxyless_builder(search_root)
    net_cfg = read_json(net_config_path)
    gene_path = resolve_gene(search_root, net_config_path, net_cfg, args.gene)
    crop_size = resolve_crop_size(net_config_path, args.crop_size)
    in_channels = int(net_cfg["blocks"][0].get("in_channels", 3))

    net = ProxylessNASNets.build_from_config(net_cfg, str(gene_path))
    checkpoint_status = load_checkpoint_if_available(net, args.checkpoint, args.device)
    net.eval().to(args.device)
    images = torch.zeros((1, in_channels, crop_size, crop_size), dtype=torch.float32, device=args.device)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else net_config_path.parent / "onnx_trt_latency"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = (
        args.onnx_path.expanduser().resolve()
        if args.onnx_path
        else output_dir / "model_fixed_b1.onnx"
    )

    with torch.no_grad():
        net(images)

    export_onnx(net, images, onnx_path)
    images_np = images.detach().cpu().numpy().astype(np.float32)

    trt_cache_dir = (
        args.trt_cache_dir.expanduser().resolve()
        if args.trt_cache_dir
        else output_dir / "trt_cache"
    )
    trt_cache_dir.mkdir(parents=True, exist_ok=True)

    providers: list[tuple[str, str, dict]] = []
    if not args.no_cpu:
        providers.append(("onnx_cpu", "CPUExecutionProvider", {}))
    if not args.no_cuda:
        providers.append(("onnx_cuda", "CUDAExecutionProvider", {"device_id": device_id}))
    if not args.no_trt:
        common_trt = {
            "device_id": device_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(trt_cache_dir),
            "trt_timing_cache_enable": True,
        }
        providers.append(("tensorrt_fp32", "TensorrtExecutionProvider", {**common_trt, "trt_fp16_enable": False}))
        providers.append(("tensorrt_fp16", "TensorrtExecutionProvider", {**common_trt, "trt_fp16_enable": True}))

    report = {
        "net_config": str(net_config_path),
        "gene": str(gene_path),
        "checkpoint": checkpoint_status,
        "onnx_path": str(onnx_path),
        "onnx_size_mb": onnx_path.stat().st_size / 1024 / 1024,
        "crop_size": crop_size,
        "in_channels": in_channels,
        "device": args.device,
        "device_name": torch.cuda.get_device_name(device_id) if torch.cuda.is_available() else None,
        "warmup": args.warmup,
        "samples": args.samples,
        "available_ort_providers": ort.get_available_providers(),
        "loaded_trt_libs": loaded_trt_libs,
        "backends": {},
    }

    print("=" * 72)
    print(f"net.config : {net_config_path}")
    print(f"gene       : {gene_path}")
    print(f"checkpoint : {checkpoint_status}")
    print(f"onnx       : {onnx_path} ({report['onnx_size_mb']:.2f} MB)")
    print(f"input      : batch=1 x {in_channels}x{crop_size}x{crop_size}, device={args.device}")
    print(f"warmup     : {args.warmup}")
    print(f"samples    : {args.samples}")
    print(f"providers  : {ort.get_available_providers()}")
    print("=" * 72)

    for label, provider, provider_options in providers:
        print(f"\n[{label}] {provider}")
        result = benchmark_provider(onnx_path, images_np, provider, provider_options, args.warmup, args.samples)
        report["backends"][label] = result
        if result.get("ok"):
            print(
                f"mean={result['mean_ms']:.6f} ms  p50={result['p50_ms']:.6f} ms  "
                f"p90={result['p90_ms']:.6f} ms  min={result['min_ms']:.6f} ms  max={result['max_ms']:.6f} ms"
            )
        else:
            print(f"failed: {result.get('error')}")

    output_json = output_dir / f"onnx_trt_latency_crop{crop_size}_{args.samples}samples.json"
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 72)
    print(f"saved_json : {output_json}")
    print("=" * 72)


if __name__ == "__main__":
    main()
