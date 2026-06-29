#!/usr/bin/env python3
"""
TensorRT 原生推理脚本（对齐 SIRST_main_all.py inference_latency）
================================================================
对齐标准（与 run_manager.inference_latency 完全一致）：
  • 输入：IRSTD-SIRST 验证集第一张真实图片（inference_loader_latency）
  • 预处理：Inference_resize=True → resize 到 256×256，Normalize([.343],[.231])
  • batch = 1
  • 计时：time.time() wall-clock，整块循环 repeat 次，除以 repeat → gpu_avg_time（秒）
    （与 SIRST_main_all 打印的 gpu_avg_time 完全对齐）
  • 同时提供 CUDA Event kernel-only 计时作为参考

流程：
  1. 加载真实图片（同 inference_loader_latency）
  2. PyTorch GPU baseline（time.time，对齐原始脚本）
  3. 从 ONNX 构建 TRT Engine（FP32 / FP16）
  4. TRT 推理：time.time（对齐） + CUDA Event（精准）

用法（在 proxylessnas/search/ 目录下）：
  conda activate proxylessnas
  python trt_infer.py                         # FP32, repeat=200
  python trt_infer.py --fp16                  # FP16
  python trt_infer.py --fp16 --repeat 500     # FP16, 500 次

依赖：tensorrt, pycuda, numpy, torch
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import os
import sys
import time

import numpy as np

# ── 路径：让 tensorrt_libs/libnvinfer.so 对 onnxruntime 可见 ─────────────────
def _add_trt_lib_to_ld() -> None:
    try:
        import tensorrt as _trt
        _site = os.path.dirname(os.path.dirname(_trt.__file__))
        _lib_dir = os.path.join(_site, "tensorrt_libs")
        if os.path.isdir(_lib_dir):
            _ld = os.environ.get("LD_LIBRARY_PATH", "")
            if _lib_dir not in _ld:
                os.environ["LD_LIBRARY_PATH"] = _lib_dir + (":" + _ld if _ld else "")
            # preload libnvinfer so onnxruntime TRT provider can find it
            for _so in sorted(glob.glob(os.path.join(_lib_dir, "libnvinfer.so*"))):
                try:
                    ctypes.CDLL(_so)
                    break
                except OSError:
                    pass
    except ImportError:
        pass

_add_trt_lib_to_ld()

# ── repo path setup ──────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
os.chdir(_script_dir)
for _p in (_repo_root, _script_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── defaults ─────────────────────────────────────────────────────────────────
_ONNX_PATH = os.path.join(
    _script_dir, "logs",
    "6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k_Retrain_8",
    "model.onnx",
)
_ENGINE_DIR = os.path.dirname(_ONNX_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_engine(onnx_path: str, engine_path: str, fp16: bool,
                 workspace_gb: float = 2.0,
                 fixed_batch: int = 1, crop_size: int = 256) -> bytes:
    """Build (or load cached) TRT engine, return serialized bytes."""
    import tensorrt as trt

    if os.path.isfile(engine_path):
        print(f"  [cache] loading engine: {engine_path}")
        with open(engine_path, "rb") as f:
            return f.read()

    logger   = trt.Logger(trt.Logger.WARNING)
    builder  = trt.Builder(logger)
    network  = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser   = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  ONNX parse error:", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(workspace_gb * (1 << 30))
    )
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  [TRT] FP16 mode enabled")
    else:
        if fp16:
            print("  [TRT] FP16 not supported on this GPU, falling back to FP32")
        else:
            print("  [TRT] FP32 mode")

    # Add optimization profile for the dynamic batch axis produced by torch.onnx.export
    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    H = W   = crop_size
    C       = network.get_input(0).shape[1]   # typically 3
    profile.set_shape(in_name,
                      min=(fixed_batch, C, H, W),
                      opt=(fixed_batch, C, H, W),
                      max=(fixed_batch, C, H, W))
    config.add_optimization_profile(profile)

    print("  Building TRT engine … (may take 1–3 min on first run)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(bytes(serialized))
    print(f"  Engine saved: {engine_path}")
    return bytes(serialized)


class TRTRunner:
    """Minimal TRT inference runner using pycuda."""

    def __init__(self, engine_bytes: bytes, device_id: int = 0):
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        self.device = cuda.Device(device_id)
        self.ctx    = self.device.make_context()

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine  = runtime.deserialize_cuda_engine(engine_bytes)
        self.trt_ctx = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # allocate buffers
        self.bindings: list = []
        self.host_inputs:   list = []
        self.device_inputs: list = []
        self.host_outputs:  list = []
        self.device_outputs: list = []

        import tensorrt as trt
        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize

            host_mem   = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device_mem = cuda.mem_alloc(nbytes)
            self.bindings.append(int(device_mem))
            self.trt_ctx.set_tensor_address(name, int(device_mem))

            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.host_inputs.append(host_mem)
                self.device_inputs.append(device_mem)
            else:
                self.host_outputs.append(host_mem)
                self.device_outputs.append(device_mem)

    def infer(self, x: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda

        np.copyto(self.host_inputs[0], x.ravel())
        cuda.memcpy_htod_async(self.device_inputs[0], self.host_inputs[0], self.stream)
        self.trt_ctx.execute_async_v3(stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.device_outputs[0], self.stream)
        self.stream.synchronize()
        return self.host_outputs[0].copy()

    def __del__(self):
        try:
            self.ctx.pop()
        except Exception:
            pass


def bench_trt_wallclock(runner: "TRTRunner", img_np: np.ndarray, repeat: int) -> float:
    """
    Aligned with run_manager.inference_latency:
    time.time() wall-clock, repeat times, return avg seconds.
    """
    import pycuda.driver as cuda

    np.copyto(runner.host_inputs[0], img_np.ravel())
    cuda.memcpy_htod_async(runner.device_inputs[0], runner.host_inputs[0], runner.stream)
    runner.stream.synchronize()

    start = time.time()
    for _ in range(repeat):
        runner.trt_ctx.execute_async_v3(stream_handle=runner.stream.handle)
        runner.stream.synchronize()
    end = time.time()
    return (end - start) / repeat


def bench_trt(runner: "TRTRunner", img_np: np.ndarray,
              warmup: int, repeat: int) -> tuple[float, float, float, float]:
    """CUDA Event kernel-only timing (accurate, for reference)."""
    import pycuda.driver as cuda

    start_ev = cuda.Event()
    end_ev   = cuda.Event()
    times: list[float] = []

    for i in range(warmup + repeat):
        np.copyto(runner.host_inputs[0], img_np.ravel())
        cuda.memcpy_htod_async(runner.device_inputs[0], runner.host_inputs[0], runner.stream)

        start_ev.record(runner.stream)
        runner.trt_ctx.execute_async_v3(stream_handle=runner.stream.handle)
        end_ev.record(runner.stream)
        end_ev.synchronize()

        cuda.memcpy_dtoh_async(runner.host_outputs[0], runner.device_outputs[0], runner.stream)
        runner.stream.synchronize()

        if i >= warmup:
            times.append(start_ev.time_till(end_ev))   # ms (GPU kernel only)

    arr = np.array(times, dtype=np.float64)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch baseline
# ─────────────────────────────────────────────────────────────────────────────

def load_real_image(args_dataset: str, args_root: str, args_split_method: str,
                    args_base_size: int, args_crop_size: int,
                    args_suffix: str, device) -> "torch.Tensor":
    """
    Load the first image from the IRSTD validation set with InferenceLoader_resize
    (Inference_resize=True), exactly as inference_loader_latency does.
    Returns a (1,3,H,W) tensor on `device`.
    """
    import torch
    from PIL import Image
    import torchvision.transforms as transforms
    from search.utils.utils import load_dataset

    dataset_dir = os.path.join(args_root, args_dataset)
    _, val_img_ids, _ = load_dataset(args_root, args_dataset, args_split_method)
    img_id = val_img_ids[0]

    img_path = os.path.join(dataset_dir, "images", img_id + args_suffix)
    img = Image.open(img_path).convert("RGB")
    # InferenceLoader_resize: resize to base_size × base_size
    img = img.resize((args_base_size, args_base_size), Image.BILINEAR)

    # IRSTD-SIRST normalization (from data_providers/SIRST.py)
    norm_mean = {"IRSTD-SIRST": [.343, .343, .343],
                 "NUAA-SIRST":  [.439, .439, .439],
                 "NUDT-SIRST":  [.423, .423, .423]}
    norm_std  = {"IRSTD-SIRST": [.231, .231, .231],
                 "NUAA-SIRST":  [.217, .217, .217],
                 "NUDT-SIRST":  [.217, .217, .217]}
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean.get(args_dataset, [.343]*3),
                             norm_std .get(args_dataset, [.231]*3)),
    ])
    tensor = tf(np.array(img)).unsqueeze(0).to(device)
    print(f"  real image : {img_id}{args_suffix}  shape={list(tensor.shape)}")
    return tensor


def bench_pytorch_wallclock(net, img_pt, repeat: int) -> float:
    """
    Aligned with run_manager.inference_latency:
    time.time() wall-clock, no warmup loop before the timed block,
    single image repeated `repeat` times, return avg seconds (gpu_avg_time).
    """
    import torch

    net.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(repeat):
            net(img_pt)
        end = time.time()
    return (end - start) / repeat


def bench_pytorch(net, dummy_pt, warmup: int, repeat: int
                  ) -> tuple[float, float, float, float]:
    """cuda.synchronize per-call timing (more accurate, for reference)."""
    import torch

    times: list[float] = []
    with torch.no_grad():
        for i in range(warmup + repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(dummy_pt)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TensorRT native inference benchmark (aligned with SIRST_main_all inference_latency)")
    parser.add_argument("--onnx",        default=_ONNX_PATH)
    parser.add_argument("--engine-dir",  default=_ENGINE_DIR)
    parser.add_argument("--fp16",        action="store_true")
    parser.add_argument("--device-id",   type=int, default=0)
    # dataset args – must match SIRST_main_all.py inference settings
    parser.add_argument("--dataset",     default="IRSTD-SIRST")
    parser.add_argument("--root",        default=os.path.join(_repo_root, "datasetyhy"))
    parser.add_argument("--split-method",default="80_20")
    parser.add_argument("--base-size",   type=int, default=256)
    parser.add_argument("--crop-size",   type=int, default=256)
    parser.add_argument("--suffix",      default=".png")
    parser.add_argument("--repeat",      type=int, default=100,
                        help="Inference_repeated (default 100, same as SIRST_main_all)")
    parser.add_argument("--warmup",      type=int, default=50,
                        help="Extra warmup before timed block (for CUDA Event ref only)")
    parser.add_argument("--no-pytorch",  action="store_true")
    args = parser.parse_args()

    import torch

    precision   = "fp16" if args.fp16 else "fp32"
    engine_name = f"model_trt_{precision}.engine"
    engine_path = os.path.join(args.engine_dir, engine_name)
    device      = torch.device(f"cuda:{args.device_id}")
    H = W       = args.base_size   # after InferenceLoader_resize

    print("=" * 66)
    print(f"  ONNX      : {args.onnx}")
    print(f"  Engine    : {engine_path}")
    print(f"  Dataset   : {args.dataset}  split={args.split_method}")
    print(f"  Input     : 1×3×{H}×{W}  (Inference_resize=True)  precision={precision.upper()}")
    print(f"  repeat={args.repeat}  (aligned with SIRST_main_all Inference_repeated)")
    print("=" * 66)

    # ── load real image (aligned: inference_loader_latency = val_img_ids[0]) ─
    print("\n[0] Loading real image from validation set …")
    img_pt = load_real_image(args.dataset, args.root, args.split_method,
                             args.base_size, args.crop_size, args.suffix, device)
    img_np = img_pt.cpu().numpy()

    # ── 1. PyTorch GPU baseline (time.time, aligned with inference_latency) ──
    if not args.no_pytorch:
        print("\n[1] PyTorch GPU baseline  (time.time, aligned with SIRST_main_all) …")
        import json
        from models import get_net_by_name

        _net_config = os.path.join(
            _script_dir, "logs",
            "6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k",
            "learned_net", "net.config",
        )
        _gene = os.path.join(
            _repo_root, "search1yhy",
            "0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new",
            "phase1_gene.txt",
        )
        _ckpt = os.path.join(
            _script_dir, "logs",
            "6,7_IRSTD-SIRST_Super_all_whole_all_31_03_2026_20_54_05_irstd1k_Retrain_8",
            "checkpoint", "model_best.pth.tar",
        )
        cfg = json.load(open(_net_config))
        net = get_net_by_name(cfg["name"]).build_from_config(cfg, _gene)
        ckpt = torch.load(_ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["state_dict"])
        net.eval().to(device)
        with torch.no_grad():
            net(img_pt)   # warmup / alloc

        # ① aligned timing (time.time, same as inference_latency)
        pt_wallclock = bench_pytorch_wallclock(net, img_pt, args.repeat)
        print(f"  gpu_avg_time (time.time)  : {pt_wallclock:.6f} s  = {pt_wallclock*1e3:.3f} ms")

        # ② cuda.synchronize per-call (reference)
        pt_mean, pt_std, pt_min, pt_max = bench_pytorch(net, img_pt, args.warmup, args.repeat)
        print(f"  CUDA-sync per-call (ref)  : mean={pt_mean:.3f}  std={pt_std:.3f}  "
              f"min={pt_min:.3f}  max={pt_max:.3f}  ms")
    else:
        pt_wallclock = pt_mean = float("nan")

    # ── 2. Build / load TRT engine ───────────────────────────────────────────
    print(f"\n[2] Building TRT engine ({precision.upper()}) …")
    engine_bytes = build_engine(args.onnx, engine_path, fp16=args.fp16,
                                fixed_batch=1, crop_size=H)

    # ── 3. TRT inference benchmark ───────────────────────────────────────────
    print(f"\n[3] TRT inference benchmark …")
    runner = TRTRunner(engine_bytes, device_id=args.device_id)
    out = runner.infer(img_np)
    print(f"  output shape: {out.shape}  (sanity ok)")

    # ① aligned timing (time.time, same as inference_latency)
    trt_wallclock = bench_trt_wallclock(runner, img_np, args.repeat)
    print(f"  gpu_avg_time (time.time)  : {trt_wallclock:.6f} s  = {trt_wallclock*1e3:.3f} ms")

    # ② CUDA Event kernel-only (reference)
    trt_mean, trt_std, trt_min, trt_max = bench_trt(runner, img_np, args.warmup, args.repeat)
    print(f"  CUDA Event kernel (ref)   : mean={trt_mean:.3f}  std={trt_std:.3f}  "
          f"min={trt_min:.3f}  max={trt_max:.3f}  ms")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  [Aligned with SIRST_main_all inference_latency: time.time wall-clock]")
    print(f"  {'Backend':<28} {'gpu_avg_time (s)':>18}  {'ms':>8}")
    print("  " + "-" * 60)
    if not args.no_pytorch:
        print(f"  {'PyTorch GPU':<28} {pt_wallclock:>18.6f}  {pt_wallclock*1e3:>8.3f}")
    print(f"  {'TensorRT ' + precision.upper():<28} {trt_wallclock:>18.6f}  {trt_wallclock*1e3:>8.3f}")
    if not np.isnan(pt_wallclock):
        print(f"\n  speed-up TRT {precision.upper()} vs PyTorch : {pt_wallclock / trt_wallclock:.2f}x")
    print("\n  [CUDA Event kernel-only timing (reference, excludes H2D/D2H)]")
    print(f"  {'Backend':<28} {'mean (ms)':>10}  {'std':>7}  {'min':>7}  {'max':>7}")
    print("  " + "-" * 60)
    if not args.no_pytorch:
        print(f"  {'PyTorch GPU':<28} {pt_mean:>10.3f}  {pt_std:>7.3f}  "
              f"{pt_min:>7.3f}  {pt_max:>7.3f}")
    print(f"  {'TensorRT ' + precision.upper():<28} {trt_mean:>10.3f}  {trt_std:>7.3f}  "
          f"{trt_min:>7.3f}  {trt_max:>7.3f}")
    print("=" * 66)


if __name__ == "__main__":
    main()
