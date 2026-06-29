#!/usr/bin/env python3
"""Rebuild stable CPU/GPU latency LUTs for the SIRST supernet.

The original RunManager LUT path uses a single time.time() sample and does not
synchronize CUDA work. This script keeps the same YAML schema, but measures each
candidate op with warmup, repeated samples, fixed CPU threading, and CUDA Events.
Values are written in seconds because LatencyEstimator multiplies them by 1000.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/proxylessnas_matplotlib")
for path in (str(REPO_ROOT), str(SEARCH_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import torch.nn.functional as F

try:
    from ruamel_yaml import YAML
except ImportError:  # pragma: no cover - compatibility fallback
    from ruamel.yaml import YAML

from models.super_nets.super_proxyless_SIRST import SuperProxylessNASNets
from utils.utils import conv_name_define, operation_choose


ParamDict = Dict[str, str]
MetricText = MutableMapping[str, str]


def parse_parameters(path: Path) -> ParamDict:
    params: ParamDict = {}
    if not path.is_file():
        return params
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":--" not in raw_line:
            continue
        key, value = raw_line.split(":--", 1)
        params[key.strip()] = value.strip()
    return params


def str_value(params: Mapping[str, str], key: str, default: str) -> str:
    value = params.get(key, default)
    return default if value in ("", "None", None) else str(value)


def int_value(params: Mapping[str, str], key: str, default: int) -> int:
    value = str_value(params, key, str(default))
    return int(float(value))


def set_reproducible(seed: int, cpu_threads: int, cudnn_benchmark: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark


def build_supernet(params: Mapping[str, str], overrides: argparse.Namespace) -> Tuple[SuperProxylessNASNets, Dict[str, Dict[str, str]], Dict[str, Any]]:
    candidates_type = overrides.candidates_type or str_value(params, "candidates_type", "whole_all")
    random_choose = str_value(params, "random_choose", "False")
    if random_choose == "True" and not overrides.allow_random_choose:
        raise ValueError(
            "parameters.txt says random_choose=True, but the sampled per-edge "
            "candidate list is not saved. Re-run with --allow-random-choose to "
            "fall back to operation_choose(candidates_type), or provide a log "
            "generated with random_choose=False."
        )

    conv_candidates = operation_choose(candidates_type)
    iterations = overrides.iterations or int_value(params, "iterations", 5)
    conv_name = conv_name_define(conv_candidates, iterations)

    gene = overrides.gene or str_value(params, "gene", "")
    if not gene:
        raise ValueError("Missing gene path. Pass --gene or use --search-log with parameters.txt.")
    gene_path = Path(gene).expanduser()
    if not gene_path.is_file():
        raise FileNotFoundError(f"gene file not found: {gene_path}")

    build_info = {
        "gene": str(gene_path),
        "iterations": iterations,
        "conv_type": overrides.conv_type or str_value(params, "conv_type", "res_add_new_new"),
        "backbone": overrides.backbone or str_value(params, "backbone", "resnet_18"),
        "in_channel": overrides.in_channel or int_value(params, "in_channel", 3),
        "num_class": overrides.num_class or int_value(params, "num_class", 1),
        "channel_num": overrides.channel_num or str_value(params, "channel_num", "two"),
        "model_name": overrides.model_name or str_value(params, "model_name", "Super_all"),
        "add_decoder": overrides.add_decoder or str_value(params, "add_decoder", "False"),
        "add_encoder0": overrides.add_encoder0 or str_value(params, "add_encoder0", "True"),
        "more_down": overrides.more_down or str_value(params, "more_down", "False"),
        "candidates_type": candidates_type,
        "num_candidates": len(conv_candidates),
    }

    net = SuperProxylessNASNets(
        conv_type=build_info["conv_type"],
        iterations=build_info["iterations"],
        backbone=build_info["backbone"],
        in_channel=build_info["in_channel"],
        conv_candidates=conv_candidates,
        n_classes=build_info["num_class"],
        channel_num=build_info["channel_num"],
        model_name=build_info["model_name"],
        gene=str(gene_path),
        add_decoder=build_info["add_decoder"],
        add_encoder0=build_info["add_encoder0"],
        more_down=build_info["more_down"],
    )
    return net, conv_name, build_info


def summarize(samples: List[float], mode: str, trim_ratio: float) -> float:
    if not samples:
        raise ValueError("empty latency sample list")
    if mode == "median":
        return statistics.median(samples)
    if mode == "mean":
        return statistics.fmean(samples)
    if mode == "min":
        return min(samples)
    if mode == "trimmed_mean":
        if not 0 <= trim_ratio < 0.5:
            raise ValueError("--trim-ratio must be in [0, 0.5)")
        ordered = sorted(samples)
        cut = int(len(ordered) * trim_ratio)
        trimmed = ordered[cut : len(ordered) - cut] if cut else ordered
        return statistics.fmean(trimmed)
    raise ValueError(f"unknown statistic: {mode}")


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_module(
    module: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeat: int,
    statistic: str,
    trim_ratio: float,
) -> Tuple[float, List[float]]:
    module.eval()
    samples: List[float] = []

    with torch.inference_mode():
        for _ in range(warmup):
            module(x)
        sync_cuda(device)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for _ in range(repeat):
                start.record()
                module(x)
                end.record()
                torch.cuda.synchronize(device)
                samples.append(start.elapsed_time(end) / 1000.0)
        else:
            for _ in range(repeat):
                start_time = time.perf_counter()
                module(x)
                samples.append(time.perf_counter() - start_time)

    return summarize(samples, statistic, trim_ratio), samples


def op_label(candidate_name: str) -> str:
    return candidate_name.split("_")[1][:-1]


def conv_desc(candidate_name: str, conv_name: Mapping[str, str]) -> Tuple[str, str, str]:
    desc = conv_name[candidate_name]
    expand = desc.split("expand:")[1].split(",")[0]
    kernel = desc.split("kernel:")[1].split(",")[0]
    group = desc.split("group:")[1].split(",")[0]
    return expand, kernel, group


def shape_str(tensor: torch.Tensor) -> Tuple[int, int, int]:
    _, c, h, w = tensor.shape
    return h, w, c


def format_entry(
    label: str,
    x_in: torch.Tensor,
    x_out: torch.Tensor,
    expand: str,
    kernel: str,
    group: str,
    latency_sec: float,
) -> str:
    h_i, w_i, c_i = shape_str(x_in)
    h_o, w_o, c_o = shape_str(x_out)
    return (
        f"{label}-input:{h_i}x{w_i}x{c_i}-output:{h_o}x{w_o}x{c_o}"
        f"-expand:{expand}-kernel:{kernel}-stride:1-group:{group},"
        f"value:{latency_sec:.12g}"
    )


def format_post_entry(x_in: torch.Tensor, x_out: torch.Tensor, latency_sec: float) -> str:
    h_i, w_i, c_i = shape_str(x_in)
    h_o, w_o, c_o = shape_str(x_out)
    return f"{h_i}x{w_i}x{c_i}-output:{h_o}x{w_o}x{c_o},value:{latency_sec:.12g}"


def prepare_block_input(
    block: torch.nn.Module,
    x: torch.Tensor,
    skip: Iterable[torch.Tensor],
    encoder_0: Any,
) -> torch.Tensor:
    _, _, h, w = x.size()
    aligned_skip = [x] + list(skip)
    skip_pre = sorted(set(block.skip_pre))
    interpolate = torch.nn.Upsample(size=(h, w), mode="bilinear", align_corners=True)
    use = [interpolate(aligned_skip[i]) for i in skip_pre]
    if encoder_0 != []:
        use.append(encoder_0)
    return torch.cat(use, dim=1)


def benchmark_mixed_edge(
    operator: torch.nn.Module,
    x: torch.Tensor,
    key_names: List[str],
    conv_name: Mapping[str, str],
    iteration: int,
    layer: int,
    device: torch.device,
    args: argparse.Namespace,
    metric_txt: MetricText,
    raw_samples: Optional[Dict[str, List[float]]],
) -> torch.Tensor:
    out_for_shape: Optional[torch.Tensor] = None

    for idx, candidate_name in enumerate(key_names):
        candidate = operator.candidate_ops[idx]
        latency_sec, samples = benchmark_module(
            candidate,
            x,
            device,
            warmup=args.warmup,
            repeat=args.repeat,
            statistic=args.statistic,
            trim_ratio=args.trim_ratio,
        )
        with torch.inference_mode():
            candidate_out = candidate(x)
        sync_cuda(device)
        if out_for_shape is None:
            out_for_shape = candidate_out

        label = op_label(candidate_name)
        expand, kernel, group = conv_desc(candidate_name, conv_name)
        yaml_key = f"{label}{iteration}_{layer}_{idx}"
        metric_txt[yaml_key] = format_entry(
            label=label,
            x_in=x,
            x_out=candidate_out,
            expand=expand,
            kernel=kernel,
            group=group,
            latency_sec=latency_sec,
        )
        if raw_samples is not None:
            raw_samples[yaml_key] = samples

    with torch.inference_mode():
        propagated = operator(x)
    sync_cuda(device)
    if out_for_shape is not None and propagated.shape != out_for_shape.shape:
        raise RuntimeError(
            f"active op output shape {tuple(propagated.shape)} differs from "
            f"candidate shape {tuple(out_for_shape.shape)} at {iteration}_{layer}"
        )
    return propagated


def rebuild_lut_for_device(
    base_net: SuperProxylessNASNets,
    conv_name: Dict[str, Dict[str, str]],
    crop_size: int,
    in_channel: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[MetricText, Optional[Dict[str, List[float]]]]:
    net = copy.deepcopy(base_net).to(device)
    net.eval()
    metric_txt: MetricText = {}
    raw_samples: Optional[Dict[str, List[float]]] = {} if args.save_samples else None

    data_shape = [1, in_channel, crop_size, crop_size]
    x = torch.zeros(data_shape, device=device)

    key_name = {
        f"iter_{idx}": list(value.keys())
        for idx, value in enumerate(conv_name.values())
    }

    with torch.inference_mode():
        x_in = x
        e_i = 0
        this_layer = args.iterations
        enc_after: List[torch.Tensor] = []
        encoder_0: List[Any] = []

        for iteration in range(args.iterations):
            enc: List[Any] = [None for _ in range(this_layer)]
            if iteration == 0:
                encoder_0 = [None for _ in range(this_layer)]

            for layer in range(this_layer):
                if layer == 0 and iteration == 0:
                    x_in = x

                block = net.encoders[e_i]
                if block.operator is not None and not block.skipped:
                    block_input = prepare_block_input(
                        block,
                        x_in,
                        enc_after if iteration != 0 else [],
                        encoder_0[layer] if iteration != 0 else [],
                    )
                    x_in = benchmark_mixed_edge(
                        operator=block.operator,
                        x=block_input,
                        key_names=key_name[f"iter_{e_i}"],
                        conv_name=conv_name[f"iter_{e_i}"],
                        iteration=iteration,
                        layer=layer,
                        device=device,
                        args=args,
                        metric_txt=metric_txt,
                        raw_samples=raw_samples,
                    )

                encoder_0[layer] = x_in if iteration == 0 else encoder_0[layer]
                if iteration != 0 and not block.skipped:
                    encoder_0[layer] = []
                if iteration == 0:
                    encoder_0[layer] = x_in

                enc[layer] = x_in
                x_in = F.max_pool2d(x_in, 2)
                e_i += 1

            this_layer -= 1
            enc_after = enc
            x_in = enc_after[0]

        post_in = x_in
        post_latency, post_samples = benchmark_module(
            net.post_transform_conv_block,
            post_in,
            device,
            warmup=args.warmup,
            repeat=args.repeat,
            statistic=args.statistic,
            trim_ratio=args.trim_ratio,
        )
        post_out = net.post_transform_conv_block(post_in)
        sync_cuda(device)
        metric_txt["Post_conv:"] = format_post_entry(post_in, post_out, post_latency)
        if raw_samples is not None:
            raw_samples["Post_conv:"] = post_samples

    return metric_txt, raw_samples


def dump_yaml(path: Path, data: MetricText) -> None:
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as handle:
        yaml.dump(data, handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild stable CPU/GPU latency LUT YAML files for ProxylessNAS SIRST search logs."
    )
    parser.add_argument("--search-log", type=Path, required=True, help="Search log directory containing parameters.txt.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write Latency_cpu.yaml / Latency_gpu.yaml.")
    parser.add_argument("--hardware", choices=("both", "cpu", "gpu"), default="both")
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index used for GPU LUT.")
    parser.add_argument("--cpu-threads", type=int, default=1, help="Fixed CPU thread count for CPU LUT.")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=200)
    parser.add_argument("--statistic", choices=("median", "mean", "trimmed_mean", "min"), default="median")
    parser.add_argument("--trim-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Allow cuDNN autotune instead of deterministic kernels.")
    parser.add_argument("--save-samples", action="store_true", help="Also save raw timing samples to JSON.")
    parser.add_argument("--allow-random-choose", action="store_true")

    parser.add_argument("--gene", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--conv-type", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--channel-num", default=None)
    parser.add_argument("--in-channel", type=int, default=None)
    parser.add_argument("--num-class", type=int, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--candidates-type", default=None)
    parser.add_argument("--add-decoder", default=None)
    parser.add_argument("--add-encoder0", default=None)
    parser.add_argument("--more-down", default=None)
    parser.add_argument("--crop-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_log = args.search_log.resolve()
    params = parse_parameters(search_log / "parameters.txt")
    if not params:
        raise FileNotFoundError(f"could not read parameters.txt under {search_log}")

    args.output_dir = (args.output_dir or search_log).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.iterations = args.iterations or int_value(params, "iterations", 5)
    crop_size = args.crop_size or int_value(params, "crop_size", 256)
    in_channel = args.in_channel or int_value(params, "in_channel", 3)

    set_reproducible(args.seed, args.cpu_threads, args.cudnn_benchmark)
    base_net, conv_name, build_info = build_supernet(params, args)

    hardware = ("cpu", "gpu") if args.hardware == "both" else (args.hardware,)
    meta: Dict[str, Any] = {
        "search_log": str(search_log),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "cpu_threads": args.cpu_threads,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "statistic": args.statistic,
        "trim_ratio": args.trim_ratio,
        "cudnn_benchmark": args.cudnn_benchmark,
        "torch_version": torch.__version__,
        "build": build_info,
        "crop_size": crop_size,
        "in_channel": in_channel,
        "devices": {},
    }

    for hw in hardware:
        if hw == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, cannot rebuild GPU LUT.")
            device = torch.device(f"cuda:{args.gpu_id}")
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
            device_name = torch.cuda.get_device_name(device)
        else:
            device = torch.device("cpu")
            device_name = f"cpu:{args.cpu_threads}threads"

        print(f"[{hw}] measuring on {device_name} ...", flush=True)
        metric_txt, raw_samples = rebuild_lut_for_device(
            base_net=base_net,
            conv_name=conv_name,
            crop_size=crop_size,
            in_channel=in_channel,
            device=device,
            args=args,
        )
        out_path = args.output_dir / f"Latency_{hw}.yaml"
        dump_yaml(out_path, metric_txt)
        meta["devices"][hw] = {
            "device_name": device_name,
            "output": str(out_path),
            "entries": len(metric_txt),
            "sum_ms": sum(float(v.split("value:")[1]) for v in metric_txt.values()) * 1000.0,
        }
        if raw_samples is not None:
            sample_path = args.output_dir / f"Latency_{hw}_samples.json"
            sample_path.write_text(json.dumps(raw_samples, indent=2), encoding="utf-8")
            meta["devices"][hw]["samples"] = str(sample_path)
        print(f"[{hw}] wrote {out_path} ({len(metric_txt)} entries)", flush=True)

    meta_path = args.output_dir / "Latency_measurement_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"metadata: {meta_path}", flush=True)


if __name__ == "__main__":
    main()
