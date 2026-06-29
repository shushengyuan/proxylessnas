#!/usr/bin/env python3
"""Retrain/evaluate a DARTS genotype on the NUAA-SIRST segmentation pipeline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import statistics
import sys
import time

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/proxylessnas_matplotlib")

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

from search.data_providers.SIRST import SIRST_DataProvider
from search.utils.utils import (
    PD_FA,
    ROCMetric,
    SoftIoULoss,
    mIoU,
    save_Ori_intensity_Pred_GT,
    make_visulization_dir,
    load_dataset_eva,
)


def load_darts_module():
    module_path = SCRIPT_DIR / "models" / "darts_sirst.py"
    spec = importlib.util.spec_from_file_location("darts_sirst_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DARTS genotype reproduction on SIRST segmentation.")
    parser.add_argument("--mode", choices=["all", "train", "eval", "latency"], default="all")
    parser.add_argument(
        "--darts-json",
        type=Path,
        default=REPO_ROOT / "compare_nas/results/darts/darts_nuaa_sirst_seed42_ep1.json",
    )
    parser.add_argument("--arch", choices=["unet_adapt", "original_dense", "same_skeleton"], default="unet_adapt")
    parser.add_argument(
        "--skeleton-config",
        type=Path,
        default=SCRIPT_DIR / "logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/net.config",
    )
    parser.add_argument(
        "--phase1-gene",
        type=Path,
        default=REPO_ROOT / "search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "logs/darts_nuaa_sirst_reproduce")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint for eval/latency.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpu", type=str, default="0", help="Comma-separated visible GPU ids, max 4 used.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, default="NUAA-SIRST")
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "datasetyhy")
    parser.add_argument("--split-method", type=str, default="50_50")
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--test-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--n-worker", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--init-lr", type=float, default=0.01)
    parser.add_argument("--fixed-lr", type=float, default=0.01)
    parser.add_argument("--lr-schedule", choices=["fixed", "cosine"], default="fixed")
    parser.add_argument("--weight-decay", type=float, default=4e-5)
    parser.add_argument("--validation-frequency", type=int, default=1)
    parser.add_argument("--print-frequency", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--darts-init-channels", type=int, default=16)
    parser.add_argument("--darts-layers", type=int, default=8)
    parser.add_argument("--darts-stem-multiplier", type=int, default=3)

    parser.add_argument("--roc-thr", type=int, default=10)
    parser.add_argument("--inference-resize", choices=["True", "False"], default="True")
    parser.add_argument("--postprocess", choices=["none"], default="none")

    parser.add_argument("--latency-device", type=str, default="cuda:0")
    parser.add_argument("--latency-warmup", type=int, default=100)
    parser.add_argument("--latency-samples", type=int, default=400)
    parser.add_argument("--sync-cuda", action="store_true", default=True)
    parser.add_argument("--no-sync-cuda", dest="sync_cuda", action="store_false")
    parser.add_argument("--cudnn-benchmark", action="store_true", default=True)
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    return parser.parse_args()


def apply_visible_gpus(gpu_arg: str) -> str:
    ids = [x.strip() for x in gpu_arg.split(",") if x.strip()]
    if len(ids) > 4:
        ids = ids[:4]
    visible = ",".join(ids) if ids else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    return visible


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalizer(dataset: str):
    if dataset == "NUAA-SIRST-Old":
        return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if dataset == "NUAA-SIRST":
        return transforms.Normalize([0.439, 0.439, 0.439], [0.217, 0.217, 0.217])
    if dataset == "NUDT-SIRST":
        return transforms.Normalize([0.423, 0.423, 0.423], [0.217, 0.217, 0.217])
    if dataset == "IRSTD-SIRST":
        return transforms.Normalize([0.343, 0.343, 0.343], [0.231, 0.231, 0.231])
    raise ValueError(f"unsupported dataset: {dataset}")


def build_data(args: argparse.Namespace):
    return SIRST_DataProvider(
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        valid_size=1,
        n_worker=args.n_worker,
        resize_scale=0.08,
        distort_color="normal",
        id_mode="TXT",
        root=str(args.root),
        split_method=args.split_method,
        base_size=args.base_size,
        crop_size=args.crop_size,
        suffix=args.suffix,
        dataset=args.dataset,
        eval_batch_size=args.eval_batch_size,
    )


def build_model(args: argparse.Namespace) -> nn.Module:
    darts_module = load_darts_module()
    if args.arch == "unet_adapt":
        model = darts_module.build_darts_sirst_from_json(args.darts_json)
    elif args.arch == "original_dense":
        model = darts_module.build_original_darts_dense_from_json(
            args.darts_json,
            init_channels=args.darts_init_channels,
            layers=args.darts_layers,
            stem_multiplier=args.darts_stem_multiplier,
        )
    elif args.arch == "same_skeleton":
        model = darts_module.build_darts_same_skeleton_from_json(
            args.darts_json,
            skeleton_config=args.skeleton_config,
            phase1_gene=args.phase1_gene,
        )
    else:
        raise ValueError(f"unsupported arch: {args.arch}")
    model.init_model()
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def checkpoint_path(args: argparse.Namespace, best: bool = False) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    name = "model_best.pth.tar" if best else "checkpoint.pth.tar"
    return args.output_dir / "checkpoint" / name


def save_checkpoint(args: argparse.Namespace, model: nn.Module, optimizer, epoch: int, best_iou: float, is_best: bool) -> None:
    ckpt_dir = args.output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / "checkpoint.pth.tar"
    state = {
        "epoch": epoch,
        "best_IOU": best_iou,
        "state_dict": unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "dataset": args.dataset,
    }
    torch.save(state, path)
    (ckpt_dir / "latest.txt").write_text(str(path) + "\n", encoding="utf-8")
    if is_best:
        torch.save({"state_dict": state["state_dict"], "best_IOU": best_iou}, ckpt_dir / "model_best.pth.tar")


def load_checkpoint(model: nn.Module, path: Path, optimizer=None) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if isinstance(state, dict):
        state = {key[len("module.") :] if key.startswith("module.") else key: val for key, val in state.items()}
    unwrap(model).load_state_dict(state)
    if optimizer is not None and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = int(checkpoint.get("epoch", -1)) + 1 if isinstance(checkpoint, dict) else 0
    best_iou = float(checkpoint.get("best_IOU", checkpoint.get("best_iou", 0.0))) if isinstance(checkpoint, dict) else 0.0
    return start_epoch, best_iou


def adjust_lr(optimizer, args: argparse.Namespace, epoch: int, batch: int, n_batch: int) -> float:
    if args.lr_schedule == "fixed":
        lr = args.fixed_lr
    elif args.lr_schedule == "cosine":
        total = max(1, args.epochs * n_batch)
        current = epoch * n_batch + batch
        lr = 0.5 * args.init_lr * (1.0 + math.cos(math.pi * current / total))
    else:
        raise ValueError(args.lr_schedule)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def validate(model: nn.Module, loader, device: torch.device, print_frequency: int, prefix: str = "Valid") -> tuple[float, float]:
    metric = mIoU(1)
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            loss = SoftIoULoss(output, labels)
            metric.update(output, labels)
            _, iou = metric.get()
            losses.append(float(loss.item()))
            if i % print_frequency == 0 or i + 1 == len(loader):
                print(f"{prefix}: [{i}/{len(loader) - 1}]\tLoss {np.mean(losses):.4f}\tvalidate_IoU {iou:.6f}")
    return float(np.mean(losses)), float(iou)


def train(args: argparse.Namespace, model: nn.Module, data, device: torch.device) -> dict:
    optimizer = torch.optim.Adagrad(model.parameters(), lr=args.init_lr, weight_decay=args.weight_decay)
    start_epoch = 0
    best_iou = 0.0
    if args.resume and checkpoint_path(args).is_file():
        start_epoch, best_iou = load_checkpoint(model, checkpoint_path(args), optimizer)

    logs_dir = args.output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    n_batch = len(data.train_data)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        metric = mIoU(1)
        losses = []
        epoch_start = time.time()
        print(f"\n------------------------------ Train epoch: {epoch + 1} ------------------------------\n")
        for i, (images, labels) in enumerate(data.train_data):
            lr = adjust_lr(optimizer, args, epoch, i, n_batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = SoftIoULoss(output, labels)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            metric.update(output.detach(), labels)
            _, train_iou = metric.get()
            losses.append(float(loss.item()))
            if i % args.print_frequency == 0 or i + 1 == n_batch:
                line = (
                    f"Train [{epoch + 1}][{i}/{n_batch - 1}]\t"
                    f"Loss {np.mean(losses):.4f}\ttrain_IoU {train_iou:.6f}\tlr {lr:.5f}"
                )
                print(line)
                with (logs_dir / "train_console.txt").open("a", encoding="utf-8") as f:
                    f.write(line + "\n")

        print(f"Time per epoch: {time.time() - epoch_start:.2f}s")
        is_best = False
        if (epoch + 1) % args.validation_frequency == 0:
            val_loss, val_iou = validate(model, data.val_data, device, args.print_frequency, prefix="Valid")
            is_best = val_iou > best_iou
            best_iou = max(best_iou, val_iou)
            line = f"Valid [{epoch + 1}/{args.epochs}]\tloss {val_loss:.4f}\tValidate_IoU {val_iou:.6f} ({best_iou:.6f})"
            print(line)
            with (logs_dir / "valid_console.txt").open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        save_checkpoint(args, model, optimizer, epoch, best_iou, is_best)

    return {"best_iou": best_iou}


def inference(args: argparse.Namespace, model: nn.Module, data, device: torch.device) -> dict:
    loader = data.inference_data_resize if args.inference_resize == "True" else data.inference_data
    metric = mIoU(1)
    roc = ROCMetric(1, args.roc_thr)
    pd_fa = PD_FA(1, 10, args.crop_size)
    target_image_path = args.output_dir / "visulization_result"
    target_dir = args.output_dir / "visulization_fuse"
    train_img_ids, val_img_ids, test_txt = load_dataset_eva(str(args.root), args.dataset, args.split_method)
    make_visulization_dir(str(target_image_path), str(target_dir))

    model.eval()
    losses = []
    with torch.no_grad():
        for idx, (images, labels, ori_img, img_id) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            output = model(images)
            loss = SoftIoULoss(output, labels)
            save_Ori_intensity_Pred_GT(
                output,
                labels,
                str(target_image_path),
                val_img_ids,
                idx,
                args.suffix,
                output.size(2),
                output.size(3),
            )
            roc.update(output, labels)
            metric.update(output, labels)
            pd_fa.update(output, labels)
            losses.append(float(loss.item()))

    _, mean_iou = metric.get()
    fa, pd = pd_fa.get(len(val_img_ids), args.crop_size)
    pd_fa_path = args.output_dir / "Pd_Fa.txt"
    with pd_fa_path.open("w", encoding="utf-8") as f:
        f.write("PD:\t" + "\t".join(str(x) for x in pd) + "\t\n")
        f.write("FA:\t" + "\t".join(str(x) for x in fa) + "\t")

    source_image_path = Path(args.root) / args.dataset / "images"
    ids = [x.strip() for x in Path(test_txt).read_text(encoding="utf-8").splitlines() if x.strip()]
    for sid in ids:
        source = source_image_path / f"{sid}{args.suffix}"
        target = target_image_path / f"{sid}{args.suffix}"
        if source.is_file() and not target.exists():
            shutil.copy(source, target)

    result = {
        "test_loss": float(np.mean(losses)),
        "validate_IoU": float(mean_iou),
        "PD": [float(x) for x in pd],
        "FA": [float(x) for x in fa],
        "pd_fa_thresholds": [i * (255 / 10) for i in range(11)],
        "pd_at_threshold_0": float(pd[0]),
        "fa_at_threshold_0": float(fa[0]),
    }
    print(f"test_loss: {result['test_loss']:.6f}")
    print(f"mean_IOU: {result['validate_IoU']:.6f}")
    print("PD:", result["PD"])
    print("FA:", result["FA"])
    return result


def benchmark_gpu_latency(args: argparse.Namespace, model: nn.Module) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for GPU latency measurement")
    device = torch.device(args.latency_device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    model = unwrap(model).to(device).eval()
    images = torch.zeros((1, 3, args.crop_size, args.crop_size), device=device)
    samples = []
    with torch.no_grad():
        for idx in range(args.latency_warmup + args.latency_samples):
            if args.sync_cuda:
                torch.cuda.synchronize(device)
            start = time.time()
            model(images)
            if args.sync_cuda:
                torch.cuda.synchronize(device)
            used_ms = (time.time() - start) * 1e3
            if idx >= args.latency_warmup:
                samples.append(float(used_ms))

    mean_ms = float(sum(samples) / len(samples))
    var_ms2 = float(sum((x - mean_ms) ** 2 for x in samples) / len(samples))
    report = {
        "model": "DARTS-SIRST segmentation adaptation",
        "darts_json": str(args.darts_json.resolve()),
        "checkpoint": str(checkpoint_path(args, best=True).resolve()) if checkpoint_path(args, best=True).is_file() else None,
        "crop_size": args.crop_size,
        "in_channels": 3,
        "device": args.latency_device,
        "device_name": torch.cuda.get_device_name(device),
        "warmup": args.latency_warmup,
        "samples": args.latency_samples,
        "sync_cuda": args.sync_cuda,
        "cudnn_benchmark": args.cudnn_benchmark,
        "mean_ms": mean_ms,
        "std_ms": math.sqrt(var_ms2),
        "var_ms2": var_ms2,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "p50_ms": statistics.median(samples),
        "samples_ms": samples,
    }
    out = args.output_dir / (
        f"gpu_latency_{args.latency_device.replace(':', '')}_crop{args.crop_size}_"
        f"{args.latency_samples}samples_sync_cudnnbench_warm{args.latency_warmup}.json"
    )
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["json_path"] = str(out)
    print(f"gpu_latency: {mean_ms:.6f} ± {report['std_ms']:.6f} ms")
    print(f"saved_json: {out}")
    return report


def write_configs(args: argparse.Namespace, model: nn.Module, visible_gpus: str) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "net.config").write_text(json.dumps(unwrap(model).save_config(), indent=2), encoding="utf-8")
    run_config = {
        "dataset": args.dataset,
        "root": str(args.root.resolve()),
        "split_method": args.split_method,
        "base_size": args.base_size,
        "crop_size": args.crop_size,
        "suffix": args.suffix,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "n_worker": args.n_worker,
        "epochs": args.epochs,
        "init_lr": args.init_lr,
        "fixed_lr": args.fixed_lr,
        "lr_schedule": args.lr_schedule,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "visible_gpus": visible_gpus,
        "darts_json": str(args.darts_json.resolve()),
        "arch": args.arch,
        "darts_init_channels": args.darts_init_channels,
        "darts_layers": args.darts_layers,
        "darts_stem_multiplier": args.darts_stem_multiplier,
        "skeleton_config": str(args.skeleton_config.resolve()),
        "phase1_gene": str(args.phase1_gene.resolve()),
    }
    (args.output_dir / "run.config").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    with (args.output_dir / "parameters.txt").open("w", encoding="utf-8") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}:--{value}\n")
        f.write(f"visible_gpus:--{visible_gpus}\n")


def write_report(args: argparse.Namespace, eval_result: dict | None, latency_result: dict | None) -> None:
    if args.arch == "original_dense":
        title = "# Original DARTS Dense Adaptation on NUAA-SIRST"
        boundary = [
            "- DARTS search artifact: `quark0/darts` genotype from local `third_party/pt.darts/searchs/nuaa_sirst_seed42_ep1`.",
            "- The cell stack follows the original DARTS augment structure: stem, normal/reduction cells at 1/3 and 2/3 depth, and two-state cell inputs.",
            "- For segmentation, only the classifier head is replaced by a 1x1 dense mask head plus bilinear upsampling.",
            "- This is stricter than the previous U-Net-like adaptation, but the local DARTS search artifact is still a binary-classification proxy search, not a native segmentation search.",
        ]
    elif args.arch == "same_skeleton":
        title = "# DARTS Same-Skeleton Baseline on NUAA-SIRST"
        boundary = [
            "- DARTS search artifact: `quark0/darts` genotype from local `third_party/pt.darts/searchs/nuaa_sirst_seed42_ep1`.",
            "- The segmentation skeleton, skip topology, and per-block channel shapes are aligned to the reference Proxyless run.",
            "- Each non-skipped skeleton block is replaced by a DARTS normal cell; MBConv/Ghost/Shuffle/Res/Spa operators are not used.",
            "- This is a fair usable DARTS-cell baseline for the segmentation setting, not a strict original classifier-DARTS model.",
        ]
    else:
        title = "# DARTS Cell Adaptation on NUAA-SIRST"
        boundary = [
            "- DARTS search artifact: `quark0/darts` genotype from local `third_party/pt.darts/searchs/nuaa_sirst_seed42_ep1`.",
            "- The local DARTS search is a binary-classification proxy search on NUAA-SIRST, not a native segmentation search.",
            "- IoU/Pd/Fa are produced by retraining the searched genotype inside a SIRST segmentation skeleton.",
        ]
    lines = [
        title,
        "",
        "## Boundary",
        "",
        *boundary,
        "",
        "## Results",
        "",
    ]
    if eval_result:
        pd0 = eval_result.get("pd_at_threshold_0")
        fa0 = eval_result.get("fa_at_threshold_0")
        lines.extend(
            [
                f"- IoU: `{eval_result['validate_IoU']:.6f}`",
                f"- Pd: `{pd0:.6f}` at threshold bin 0",
                f"- Fa: `{fa0:.12f}` at threshold bin 0",
                f"- Full PD/FA curve: `{args.output_dir / 'Pd_Fa.txt'}`",
            ]
        )
    if latency_result:
        lines.append(
            f"- GPU latency: `{latency_result['mean_ms']:.3f} ± {latency_result['std_ms']:.3f} ms` "
            f"({latency_result['samples']} samples, warmup {latency_result['warmup']})"
        )
        lines.append(f"- GPU latency samples: `{latency_result['json_path']}`")
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "cd /home/intern/proxylessnas",
            (
                "/home/intern/anaconda3/envs/new_env/bin/python search/darts_sirst_reproduce.py "
                f"--mode all --arch {args.arch} --gpu {args.gpu} --epochs {args.epochs} --train-batch-size {args.train_batch_size} "
                f"--validation-frequency {args.validation_frequency} "
                f"--output-dir {args.output_dir}"
            ),
            "```",
        ]
    )
    (args.output_dir / "darts_sirst_reproduction.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    visible_gpus = apply_visible_gpus(args.gpu)
    set_seed(args.seed)
    cudnn.benchmark = args.cudnn_benchmark

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = None if args.mode == "latency" else build_data(args)
    model = build_model(args)
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.mode in {"all", "train", "eval"}:
        model = nn.DataParallel(model)
    model.to(device)
    write_configs(args, model, visible_gpus)

    eval_result = None
    latency_result = None
    if args.mode in {"all", "train"}:
        train(args, model, data, device)

    if args.mode in {"all", "eval"}:
        ckpt = checkpoint_path(args, best=True)
        if not ckpt.is_file():
            ckpt = checkpoint_path(args, best=False)
        load_checkpoint(model, ckpt)
        eval_result = inference(args, model, data, device)
        output = {"test_loss": f"{eval_result['test_loss']:.6f}", "validate_IoU": f"{eval_result['validate_IoU']:.6f}"}
        (args.output_dir / "output").write_text(json.dumps(output, indent=4), encoding="utf-8")

    if args.mode in {"all", "latency"}:
        ckpt = checkpoint_path(args, best=True)
        if args.mode == "latency" and args.checkpoint is not None:
            ckpt = args.checkpoint
        if ckpt.is_file():
            load_checkpoint(model, ckpt)
        latency_result = benchmark_gpu_latency(args, model)

    write_report(args, eval_result, latency_result)


if __name__ == "__main__":
    main()
