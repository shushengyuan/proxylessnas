#!/usr/bin/env python3
"""Benchmark an ONNX model with ONNX Runtime CUDA IO binding."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import statistics
import time

import numpy as np


def preload_cuda_libs() -> list[str]:
    """Make ORT CUDA EP dependencies visible in conda/pip CUDA layouts."""

    loaded: list[str] = []
    candidates: list[Path] = []
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

    try:
        import nvidia.cudnn

        candidates.append(Path(nvidia.cudnn.__file__).resolve().parent / "lib")
    except Exception:
        pass

    for env_path in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if env_path:
            candidates.append(Path(env_path))

    for lib_dir in candidates:
        if not lib_dir.is_dir():
            continue
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if str(lib_dir) not in ld_path.split(":"):
            os.environ["LD_LIBRARY_PATH"] = str(lib_dir) + (":" + ld_path if ld_path else "")
        for pattern in ("libcudnn.so.9", "libcublas.so*", "libcublasLt.so*"):
            for so_path in sorted(lib_dir.glob(pattern)):
                try:
                    ctypes.CDLL(str(so_path))
                    loaded.append(str(so_path))
                    break
                except OSError:
                    pass
    return loaded


def percentile(values: list[float], p: float) -> float:
    xs = sorted(values)
    k = (len(xs) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c - k) + xs[c] * (k - f)


def summarize(samples: list[float]) -> dict:
    mean_ms = sum(samples) / len(samples)
    var_ms2 = sum((x - mean_ms) ** 2 for x in samples) / len(samples)
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded_cuda_libs = preload_cuda_libs()

    import onnxruntime as ort

    onnx_path = args.onnx_path.expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    provider_options = {"device_id": args.device_id}
    if args.cuda_graph:
        provider_options["enable_cuda_graph"] = "1"

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 2
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=opts,
        providers=[("CUDAExecutionProvider", provider_options)],
    )
    input_name = session.get_inputs()[0].name
    output_names = [node.name for node in session.get_outputs()]

    images_np = np.zeros((1, args.in_channels, args.crop_size, args.crop_size), dtype=np.float32)
    input_ort = ort.OrtValue.ortvalue_from_numpy(images_np, "cuda", args.device_id)

    io_binding = session.io_binding()
    io_binding.bind_input(
        name=input_name,
        device_type="cuda",
        device_id=args.device_id,
        element_type=np.float32,
        shape=images_np.shape,
        buffer_ptr=input_ort.data_ptr(),
    )
    for output_name in output_names:
        io_binding.bind_output(output_name, "cuda", args.device_id)

    times: list[float] = []
    for idx in range(args.warmup + args.samples):
        start = time.perf_counter()
        session.run_with_iobinding(io_binding)
        used_ms = (time.perf_counter() - start) * 1e3
        if idx >= args.warmup:
            times.append(used_ms)

    report = {
        "onnx_path": str(onnx_path),
        "provider": "CUDAExecutionProvider",
        "provider_options": provider_options,
        "actual_providers": session.get_providers(),
        "input_shape": list(images_np.shape),
        "output_names": output_names,
        "warmup": args.warmup,
        "samples": args.samples,
        "io_binding": True,
        "cuda_graph": args.cuda_graph,
        "loaded_cuda_libs": loaded_cuda_libs,
        **summarize(times),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"onnx: {onnx_path}")
    print(f"io_binding: True")
    print(f"cuda_graph: {args.cuda_graph}")
    print(f"mean_ms: {report['mean_ms']:.6f}")
    print(f"std_ms: {report['std_ms']:.6f}")
    print(f"p50_ms: {report['p50_ms']:.6f}")
    print(f"saved_json: {args.output}")


if __name__ == "__main__":
    main()
