#!/usr/bin/env python3
"""
统一推理 Benchmark 脚本
======================
覆盖：PyTorch GPU / ONNX CPU / ONNX CUDA / TRT FP32 / TRT FP16

对齐标准（与 run_manager.inference_latency 完全一致）：
  • 输入：真实验证集第一张图（inference_loader_latency → val_img_ids[0]）
  • 预处理：Inference_resize=True → resize(base_size) + Normalize
  • batch=1, repeat=100, time.time() wall-clock → gpu_avg_time
  • 同时提供 CUDA Event / cuda.synchronize 精准计时（参考）

用法：
  cd proxylessnas/search
  conda activate proxylessnas

  # 指定 search log 目录（脚本自动找 _Retrain / net.config / gene / dataset）
  python bench_all.py --search-log logs/0,1_NUAA-SIRST_Super_all_whole_all_10_04_2026_01_42_51

  # 也可以明确指定 retrain dir
  python bench_all.py \\
      --search-log logs/0,1_NUAA-SIRST_Super_all_whole_all_10_04_2026_01_42_51 \\
      --retrain-log logs/0,1_NUAA-SIRST_Super_all_whole_all_10_04_2026_01_42_51_Retrain

  # 跳过 TRT（无 tensorrt 时）
  python bench_all.py --search-log ... --no-trt

依赖：torch, onnx, onnxruntime-gpu, tensorrt, pycuda
"""

from __future__ import annotations

import argparse
import ctypes
import glob
import json
import os
import sys
import time
import warnings

import numpy as np

# ── path setup ───────────────────────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root  = os.path.dirname(_script_dir)
os.chdir(_script_dir)
for _p in (_repo_root, _script_dir):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── preload libnvinfer so ORT TRT provider can find it ───────────────────────
def _preload_trt_libs() -> bool:
    try:
        import tensorrt as trt
        site = os.path.dirname(os.path.dirname(trt.__file__))
        lib_dir = os.path.join(site, "tensorrt_libs")
        if os.path.isdir(lib_dir):
            ld = os.environ.get("LD_LIBRARY_PATH", "")
            if lib_dir not in ld:
                os.environ["LD_LIBRARY_PATH"] = lib_dir + (":" + ld if ld else "")
            for so in sorted(glob.glob(os.path.join(lib_dir, "libnvinfer.so*"))):
                try:
                    ctypes.CDLL(so)
                    return True
                except OSError:
                    pass
    except ImportError:
        pass
    return False


_HAS_TRT = _preload_trt_libs()


# ─────────────────────────────────────────────────────────────────────────────
# Auto-detect helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_parameters(log_dir: str) -> dict:
    p = os.path.join(log_dir, "parameters.txt")
    result: dict = {}
    if not os.path.isfile(p):
        return result
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":--" in line:
                k, _, v = line.partition(":--")
                result[k.strip()] = v.strip()
    return result


def _resolve_paths(args) -> dict:
    """
    Given --search-log, auto-fill: retrain_log, net_config, gene, checkpoint,
    dataset, split_method, base_size, suffix, root.
    CLI overrides take precedence.
    """
    search_log = os.path.abspath(args.search_log)

    # retrain dir: explicit > search_log + "_Retrain" (suffix _N also ok)
    if args.retrain_log:
        retrain_log = os.path.abspath(args.retrain_log)
    else:
        candidate = search_log + "_Retrain"
        if not os.path.isdir(candidate):
            # try numbered suffixes
            for suf in [f"_Retrain_{i}" for i in range(1, 20)]:
                if os.path.isdir(search_log + suf):
                    candidate = search_log + suf
                    break
        retrain_log = candidate

    net_config  = os.path.join(search_log, "learned_net", "net.config")
    checkpoint  = os.path.join(retrain_log, "checkpoint", "model_best.pth.tar")

    # gene from parameters.txt (search log)
    params = _read_parameters(search_log)
    gene = args.gene or params.get("gene", "")
    if gene and not os.path.isfile(gene):
        # try relative to repo root
        candidate_g = os.path.join(_repo_root, gene.lstrip("/"))
        if os.path.isfile(candidate_g):
            gene = candidate_g

    # dataset / split / size from retrain run.config, fallback to parameters.txt
    run_cfg: dict = {}
    rc_path = os.path.join(retrain_log, "run.config")
    if os.path.isfile(rc_path):
        run_cfg = json.load(open(rc_path, encoding="utf-8"))

    dataset      = args.dataset      or run_cfg.get("dataset")      or params.get("dataset",      "NUAA-SIRST")
    split_method = args.split_method or run_cfg.get("split_method") or params.get("split_method", "50_50")
    base_size    = args.base_size    or int(run_cfg.get("base_size", 0)) or int(params.get("base_size", 256))
    suffix       = args.suffix       or run_cfg.get("suffix")       or params.get("suffix",       ".png")
    root         = args.root         or run_cfg.get("root")         or params.get("root",
                                                                                  os.path.join(_repo_root, "datasetyhy"))
    root = root.replace("/home/intern/proxylessnas/search/..", _repo_root)

    onnx_path   = os.path.join(retrain_log, "model.onnx")
    engine_dir  = retrain_log

    return dict(
        search_log=search_log, retrain_log=retrain_log,
        net_config=net_config, gene=gene, checkpoint=checkpoint,
        dataset=dataset, split_method=split_method,
        base_size=base_size, suffix=suffix, root=root,
        onnx_path=onnx_path, engine_dir=engine_dir,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image loader (aligned with InferenceLoader_resize + inference_loader_latency)
# ─────────────────────────────────────────────────────────────────────────────

def load_real_image(dataset: str, root: str, split_method: str,
                    base_size: int, suffix: str, device) -> tuple:
    """Returns (img_pt [1,3,H,W] on device, img_np [1,3,H,W] float32)."""
    import torch
    from PIL import Image
    import torchvision.transforms as T
    from search.utils.utils import load_dataset

    dataset_dir = os.path.join(root, dataset)
    _, val_img_ids, _ = load_dataset(root, dataset, split_method)
    img_id = val_img_ids[0]
    img_path = os.path.join(dataset_dir, "images", img_id + suffix)

    img = Image.open(img_path).convert("RGB")
    img = img.resize((base_size, base_size), Image.BILINEAR)  # InferenceLoader_resize

    NORM = {
        "IRSTD-SIRST":   ([.343]*3, [.231]*3),
        "NUAA-SIRST":    ([.439]*3, [.217]*3),
        "NUAA-SIRST-Old":([.485,.456,.406], [.229,.224,.225]),
        "NUDT-SIRST":    ([.423]*3, [.217]*3),
    }
    mean, std = NORM.get(dataset, ([.439]*3, [.217]*3))
    tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    img_pt = tf(np.array(img)).unsqueeze(0).to(device)
    img_np = img_pt.cpu().numpy().astype(np.float32)
    print(f"  real image : {img_id}{suffix}  shape={list(img_pt.shape)}")
    return img_pt, img_np


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch
# ─────────────────────────────────────────────────────────────────────────────

def load_pytorch_model(net_config: str, gene: str, checkpoint: str, device):
    import torch
    from models import get_net_by_name

    cfg = json.load(open(net_config, encoding="utf-8"))
    net = get_net_by_name(cfg["name"]).build_from_config(cfg, gene)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    return net.eval().to(device)


def bench_pytorch_wallclock(net, img_pt, repeat: int) -> float:
    """time.time wall-clock, aligned with inference_latency. Returns avg seconds."""
    import torch
    net.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(repeat):
            net(img_pt)
        return (time.time() - t0) / repeat


def bench_pytorch_event(net, img_pt, warmup: int, repeat: int
                        ) -> tuple[float, float, float, float]:
    """cuda.synchronize per-call (ms). Returns (mean, std, min, max)."""
    import torch
    times = []
    with torch.no_grad():
        for i in range(warmup + repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net(img_pt)
            torch.cuda.synchronize()
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


def bench_pytorch_cpu_wallclock(net, img_cpu, repeat: int) -> float:
    """Same as bench_pytorch_wallclock but CPU — raw PyTorch, no ONNX."""
    import torch

    net.eval()
    with torch.no_grad():
        t0 = time.time()
        for _ in range(repeat):
            net(img_cpu)
        return (time.time() - t0) / repeat


def bench_pytorch_cpu_perrun(net, img_cpu, warmup: int, repeat: int
                             ) -> tuple[float, float, float, float]:
    """Per-inference perf_counter on CPU (ms), no CUDA sync."""
    import torch

    times: list[float] = []
    with torch.no_grad():
        for i in range(warmup + repeat):
            t0 = time.perf_counter()
            net(img_cpu)
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times, dtype=np.float64)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


# ─────────────────────────────────────────────────────────────────────────────
# ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(net, img_pt, onnx_path: str) -> None:
    import torch, onnx as ox

    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            net, img_pt, onnx_path, opset_version=17,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            do_constant_folding=True,
        )
    ox.checker.check_model(ox.load(onnx_path))
    mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  ONNX saved : {onnx_path}  ({mb:.2f} MB)")


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

def bench_onnx_wallclock(onnx_path: str, img_np: np.ndarray,
                         provider: str, repeat: int) -> float | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                        providers=[provider])
        if provider not in sess.get_providers():
            return None
    except Exception:
        return None

    name = sess.get_inputs()[0].name
    # warmup
    for _ in range(10):
        sess.run(None, {name: img_np})
    t0 = time.time()
    for _ in range(repeat):
        sess.run(None, {name: img_np})
    return (time.time() - t0) / repeat


def bench_onnx_perrun(onnx_path: str, img_np: np.ndarray,
                      provider: str, warmup: int, repeat: int
                      ) -> tuple[float, float, float, float] | None:
    try:
        import onnxruntime as ort
    except ImportError:
        return None

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    provider_opts = {}
    if "Tensorrt" in provider:
        provider_opts = {"trt_max_workspace_size": 1 << 30}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                        providers=[(provider, provider_opts)])
        if provider not in sess.get_providers():
            return None
    except Exception:
        return None

    name = sess.get_inputs()[0].name
    times = []
    for i in range(warmup + repeat):
        t0 = time.perf_counter()
        sess.run(None, {name: img_np})
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    return float(a.mean()), float(a.std()), float(a.min()), float(a.max())


# ─────────────────────────────────────────────────────────────────────────────
# TensorRT native
# ─────────────────────────────────────────────────────────────────────────────

def build_trt_engine(onnx_path: str, engine_path: str, fp16: bool,
                     fixed_batch: int = 1, crop_size: int = 256,
                     workspace_gb: float = 2.0) -> bytes | None:
    try:
        import tensorrt as trt
    except ImportError:
        return None

    if os.path.isfile(engine_path):
        print(f"    [cache] {engine_path}")
        return open(engine_path, "rb").read()

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  parse error:", parser.get_error(i))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 int(workspace_gb * (1 << 30)))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("    FP16 enabled")
    else:
        print("    FP32 mode" + (" (GPU no FP16)" if fp16 else ""))

    profile = builder.create_optimization_profile()
    in_name = network.get_input(0).name
    C = network.get_input(0).shape[1]
    H = W = crop_size
    profile.set_shape(in_name,
                      min=(fixed_batch, C, H, W),
                      opt=(fixed_batch, C, H, W),
                      max=(fixed_batch, C, H, W))
    config.add_optimization_profile(profile)

    print("    Building … (first run ~1-3 min)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        return None
    data = bytes(serialized)
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"    Saved: {engine_path}")
    return data


class TRTRunner:
    def __init__(self, engine_bytes: bytes, device_id: int = 0):
        import pycuda.driver as cuda
        import tensorrt as trt

        cuda.init()
        self.ctx    = cuda.Device(device_id).make_context()
        runtime     = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.trt_ctx = self.engine.create_execution_context()
        self.stream  = cuda.Stream()
        self.host_in: list = []
        self.dev_in:  list = []
        self.host_out: list = []
        self.dev_out:  list = []

        for i in range(self.engine.num_io_tensors):
            name  = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            nb    = int(np.prod(shape)) * np.dtype(dtype).itemsize
            hmem  = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dmem  = cuda.mem_alloc(nb)
            self.trt_ctx.set_tensor_address(name, int(dmem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.host_in.append(hmem); self.dev_in.append(dmem)
            else:
                self.host_out.append(hmem); self.dev_out.append(dmem)

    def _upload(self, img_np):
        import pycuda.driver as cuda
        np.copyto(self.host_in[0], img_np.ravel())
        cuda.memcpy_htod_async(self.dev_in[0], self.host_in[0], self.stream)
        self.stream.synchronize()

    def bench_wallclock(self, img_np: np.ndarray, repeat: int) -> float:
        import pycuda.driver as cuda
        self._upload(img_np)
        t0 = time.time()
        for _ in range(repeat):
            self.trt_ctx.execute_async_v3(stream_handle=self.stream.handle)
            self.stream.synchronize()
        return (time.time() - t0) / repeat

    def bench_event(self, img_np: np.ndarray, warmup: int, repeat: int
                    ) -> tuple[float, float, float, float]:
        import pycuda.driver as cuda
        self._upload(img_np)
        se, ee = cuda.Event(), cuda.Event()
        times = []
        for i in range(warmup + repeat):
            se.record(self.stream)
            self.trt_ctx.execute_async_v3(stream_handle=self.stream.handle)
            ee.record(self.stream); ee.synchronize()
            if i >= warmup:
                times.append(se.time_till(ee))
        a = np.array(times, dtype=np.float64)
        return float(a.mean()), float(a.std()), float(a.min()), float(a.max())

    def __del__(self):
        try: self.ctx.pop()
        except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# pretty print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _row_wc(label, wc_s):
    if wc_s is None or (isinstance(wc_s, float) and np.isnan(wc_s)):
        return f"  {label:<30}  {'N/A':>10}  {'N/A':>8}"
    return f"  {label:<30}  {wc_s:>10.6f}  {wc_s*1e3:>8.3f}"


def _row_ev(label, ev):
    if ev is None:
        return f"  {label:<30}  {'N/A':>8}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}"
    m, s, mn, mx = ev
    return f"  {label:<30}  {m:>8.3f}  {s:>7.3f}  {mn:>7.3f}  {mx:>7.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified PyTorch / ONNX / TensorRT inference benchmark")
    parser.add_argument("--search-log",   required=True,
                        help="Search log dir (e.g. logs/0,1_NUAA-...)")
    parser.add_argument("--retrain-log",  default=None,
                        help="Retrain log dir (default: search_log + _Retrain)")
    parser.add_argument("--gene",         default=None)
    parser.add_argument("--dataset",      default=None)
    parser.add_argument("--split-method", default=None)
    parser.add_argument("--base-size",    type=int, default=None)
    parser.add_argument("--suffix",       default=None)
    parser.add_argument("--root",         default=None)
    parser.add_argument("--device-id",    type=int, default=0)
    parser.add_argument("--repeat",       type=int, default=100,
                        help="Inference_repeated (default 100)")
    parser.add_argument("--warmup",       type=int, default=50,
                        help="Warmup for per-run timing (default 50)")
    parser.add_argument("--no-trt",       action="store_true")
    parser.add_argument("--no-pytorch",   action="store_true")
    args = parser.parse_args()

    import torch
    device = torch.device(f"cuda:{args.device_id}")
    cfg = _resolve_paths(args)

    print("\n" + "=" * 70)
    print("  Benchmark configuration")
    print("  " + "-" * 66)
    for k in ("search_log","retrain_log","net_config","gene","checkpoint",
              "dataset","split_method","base_size","suffix","root"):
        print(f"  {k:<16}: {cfg[k]}")
    print(f"  {'device':<16}: {device}  repeat={args.repeat}  warmup={args.warmup}")
    print("=" * 70)

    # sanity checks
    for key in ("net_config", "checkpoint"):
        if not os.path.isfile(cfg[key]):
            raise SystemExit(f"File not found: {cfg[key]}")
    if not cfg["gene"] or not os.path.isfile(cfg["gene"]):
        raise SystemExit(f"gene not found: {cfg['gene']}")

    # ── load real image ───────────────────────────────────────────────────────
    print("\n[0] Loading real validation image …")
    img_pt, img_np = load_real_image(
        cfg["dataset"], cfg["root"], cfg["split_method"],
        cfg["base_size"], cfg["suffix"], device)

    results_wc: dict[str, float | None] = {}
    results_ev: dict[str, tuple | None] = {}

    # ── PyTorch ───────────────────────────────────────────────────────────────
    net = None
    if not args.no_pytorch:
        print("\n[1] PyTorch GPU …")
        net = load_pytorch_model(cfg["net_config"], cfg["gene"],
                                 cfg["checkpoint"], device)
        with torch.no_grad():
            net(img_pt)   # warmup alloc

        wc = bench_pytorch_wallclock(net, img_pt, args.repeat)
        ev = bench_pytorch_event(net, img_pt, args.warmup, args.repeat)
        results_wc["PyTorch GPU"] = wc
        results_ev["PyTorch GPU"] = ev
        print(f"  gpu_avg_time : {wc:.6f} s  = {wc*1e3:.3f} ms")
        print(f"  sync/call    : mean={ev[0]:.3f}  std={ev[1]:.3f}  "
              f"min={ev[2]:.3f}  max={ev[3]:.3f}  ms")

        # Raw PyTorch on CPU (same image tensor moved to CPU; model .cpu() then back to GPU for ONNX)
        print("\n[1b] PyTorch CPU (raw torch, aligned timing with [1]) …")
        img_cpu = img_pt.detach().cpu()
        net_cpu = net.cpu()
        wc_cpu = bench_pytorch_cpu_wallclock(net_cpu, img_cpu, args.repeat)
        ev_cpu = bench_pytorch_cpu_perrun(net_cpu, img_cpu, args.warmup, args.repeat)
        results_wc["PyTorch CPU"] = wc_cpu
        results_ev["PyTorch CPU"] = ev_cpu
        print(f"  avg_time     : {wc_cpu:.6f} s  = {wc_cpu*1e3:.3f} ms  (time.time / repeat, same as GPU line)")
        print(f"  per-call     : mean={ev_cpu[0]:.3f}  std={ev_cpu[1]:.3f}  "
              f"min={ev_cpu[2]:.3f}  max={ev_cpu[3]:.3f}  ms  (perf_counter)")
        net = net_cpu.to(device)

    # ── ONNX export ───────────────────────────────────────────────────────────
    print("\n[2] ONNX export …")
    if net is None:
        net = load_pytorch_model(cfg["net_config"], cfg["gene"],
                                 cfg["checkpoint"], device)
        with torch.no_grad():
            net(img_pt)
    if not os.path.isfile(cfg["onnx_path"]):
        export_onnx(net, img_pt, cfg["onnx_path"])
    else:
        print(f"  [cache] {cfg['onnx_path']}")

    # ── ONNX CPU ──────────────────────────────────────────────────────────────
    print("\n[3] ONNX CPU …")
    wc = bench_onnx_wallclock(cfg["onnx_path"], img_np,
                              "CPUExecutionProvider", args.repeat)
    ev = bench_onnx_perrun(cfg["onnx_path"], img_np,
                           "CPUExecutionProvider", args.warmup, args.repeat)
    results_wc["ONNX CPU"] = wc
    results_ev["ONNX CPU"] = ev
    if wc is not None:
        print(f"  gpu_avg_time : {wc:.6f} s  = {wc*1e3:.3f} ms")
        if ev: print(f"  per-call     : mean={ev[0]:.3f}  std={ev[1]:.3f}  "
                     f"min={ev[2]:.3f}  max={ev[3]:.3f}  ms")

    # ── ONNX CUDA ─────────────────────────────────────────────────────────────
    print("\n[4] ONNX CUDA …")
    wc = bench_onnx_wallclock(cfg["onnx_path"], img_np,
                              "CUDAExecutionProvider", args.repeat)
    ev = bench_onnx_perrun(cfg["onnx_path"], img_np,
                           "CUDAExecutionProvider", args.warmup, args.repeat)
    results_wc["ONNX CUDA"] = wc
    results_ev["ONNX CUDA"] = ev
    if wc is not None:
        print(f"  gpu_avg_time : {wc:.6f} s  = {wc*1e3:.3f} ms")
        if ev: print(f"  per-call     : mean={ev[0]:.3f}  std={ev[1]:.3f}  "
                     f"min={ev[2]:.3f}  max={ev[3]:.3f}  ms")

    # ── TensorRT ──────────────────────────────────────────────────────────────
    if not args.no_trt:
        for fp16 in (False, True):
            prec  = "FP16" if fp16 else "FP32"
            label = f"TRT {prec}"
            ename = f"model_trt_{'fp16' if fp16 else 'fp32'}.engine"
            epath = os.path.join(cfg["engine_dir"], ename)

            print(f"\n[{'5' if not fp16 else '6'}] TensorRT {prec} …")
            eng = build_trt_engine(cfg["onnx_path"], epath, fp16=fp16,
                                   fixed_batch=1, crop_size=cfg["base_size"])
            if eng is None:
                print("  skipped (build failed or TRT not available)")
                results_wc[label] = None
                results_ev[label] = None
                continue

            runner = TRTRunner(eng, device_id=args.device_id)
            wc = runner.bench_wallclock(img_np, args.repeat)
            ev = runner.bench_event(img_np, args.warmup, args.repeat)
            results_wc[label] = wc
            results_ev[label] = ev
            print(f"  gpu_avg_time : {wc:.6f} s  = {wc*1e3:.3f} ms")
            print(f"  CUDA Event   : mean={ev[0]:.3f}  std={ev[1]:.3f}  "
                  f"min={ev[2]:.3f}  max={ev[3]:.3f}  ms")
            del runner

    # ── Summary ───────────────────────────────────────────────────────────────
    base_wc = results_wc.get("PyTorch GPU")

    print("\n" + "=" * 70)
    print("  SUMMARY  (aligned with inference_latency: time.time wall-clock)")
    print(f"  {'Backend':<30}  {'avg (s)':>10}  {'ms':>8}  {'speedup':>8}")
    print("  " + "-" * 66)
    for name, wc in results_wc.items():
        if wc is None:
            sp = "N/A"
            print(f"  {name:<30}  {'N/A':>10}  {'N/A':>8}  {sp:>8}")
        else:
            sp = f"{base_wc/wc:.2f}x" if (base_wc and not np.isnan(base_wc)) else "-"
            print(f"  {name:<30}  {wc:>10.6f}  {wc*1e3:>8.3f}  {sp:>8}")

    print()
    print("  [Per-run timing (ms): GPU = cuda.synchronize + perf_counter; "
          "CPU torch/ONNX = perf_counter only]")
    print(f"  {'Backend':<30}  {'mean':>8}  {'std':>7}  {'min':>7}  {'max':>7}")
    print("  " + "-" * 66)
    for name, ev in results_ev.items():
        print(_row_ev(name, ev))
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
