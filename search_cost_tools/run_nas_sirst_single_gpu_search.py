#!/usr/bin/env python3
import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU Nas-SIRST search cost runner")
    parser.add_argument("--repo", default="/home/intern/nas/Nas-SIRST-UNet-inference")
    parser.add_argument("--data-root", default="/home/intern/proxylessnas/datasetyhy")
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument("--split-method", default="50_50")
    parser.add_argument("--suffix", default=".png")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--epochs", type=int, default=300, help="Safety cap, not the search stopping point")
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20, help="Stop after this many unchanged genotype checks")
    parser.add_argument("--disable-stability-stop", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--init-channels", type=int, default=16)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--meta-node-num", type=int, default=3)
    parser.add_argument("--alpha-begin", type=int, default=10)
    parser.add_argument("--train-portion", type=float, default=0.5)
    parser.add_argument("--model-lr", type=float, default=1e-3)
    parser.add_argument("--model-weight-decay", type=float, default=5e-4)
    parser.add_argument("--arch-lr", type=float, default=3e-4)
    parser.add_argument("--arch-weight-decay", type=float, default=5e-3)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/home/intern/proxylessnas/search_cost_tools/runs/nas_sirst_single_gpu_search",
    )
    return parser.parse_args()


def setup_imports(repo):
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    return repo


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unique_params(params):
    seen = set()
    out = []
    for param in params:
        if id(param) in seen:
            continue
        seen.add(id(param))
        out.append(param)
    return out


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main():
    args = parse_args()
    start_wall = time.time()
    start_dt = datetime.now()

    repo = setup_imports(args.repo)
    from adabound import AdaBound
    from model.load_param_data import load_dataset
    from search.backbone.nas_unet_search import NasUnetSearch
    from util.loss.loss import SoftIoULoss
    from util.utils_SIRST import TrainSetLoader

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU search cost measurement")

    output_root = Path(args.output_dir)
    run_name = start_dt.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    summary_path = run_dir / "summary.json"

    def log(message):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    log("Nas-SIRST single-GPU search started")
    log(f"repo={repo}")
    log(f"args={vars(args)}")
    log(f"cuda_visible_devices={args.gpu}")
    log(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    train_ids, _, _ = load_dataset(args.data_root, args.dataset, args.split_method)
    rng = random.Random(args.seed)
    train_ids = list(train_ids)
    rng.shuffle(train_ids)
    split = max(1, min(len(train_ids) - 1, int(len(train_ids) * args.train_portion)))
    weight_ids = train_ids[:split]
    arch_ids = train_ids[split:]
    log(f"dataset={args.dataset} split={args.split_method} weight_ids={len(weight_ids)} arch_ids={len(arch_ids)}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.439, 0.439, 0.439], [0.217, 0.217, 0.217]),
        ]
    )
    dataset_dir = Path(args.data_root) / args.dataset
    weight_set = TrainSetLoader(
        str(dataset_dir),
        weight_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=transform,
        suffix=args.suffix,
    )
    arch_set = TrainSetLoader(
        str(dataset_dir),
        arch_ids,
        base_size=args.base_size,
        crop_size=args.crop_size,
        transform=transform,
        suffix=args.suffix,
    )
    weight_loader = DataLoader(
        weight_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    arch_loader = DataLoader(
        arch_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    if len(weight_loader) == 0 or len(arch_loader) == 0:
        raise RuntimeError("DataLoader has zero batches; reduce batch size")

    model = NasUnetSearch(
        3,
        args.init_channels,
        1,
        args.depth,
        meta_node_num=args.meta_node_num,
        use_sharing=True,
        double_down_channel=False,
        use_softmax_head=False,
        multi_gpus=False,
        device=device,
    ).to(device)

    arch_param_ids = {id(param) for param in model.arch_parameters()}
    weight_params = [param for param in model.parameters() if id(param) not in arch_param_ids]
    arch_params = unique_params(model.arch_parameters())
    weight_optimizer = AdaBound(
        weight_params,
        lr=args.model_lr,
        weight_decay=args.model_weight_decay,
    )
    arch_optimizer = torch.optim.Adam(
        arch_params,
        lr=args.arch_lr,
        weight_decay=args.arch_weight_decay,
    )

    log(f"weight_batches={len(weight_loader)} arch_batches={len(arch_loader)}")
    log(f"weight_params={sum(p.numel() for p in weight_params)} arch_params={sum(p.numel() for p in arch_params)}")

    arch_iter = iter(arch_loader)
    history = []
    last_genotype = None
    stable_epochs = 0
    stop_reason = "max_epochs_reached"
    completed_epochs = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        arch_loss_sum = 0.0
        arch_steps = 0

        for images, masks in weight_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if epoch >= args.alpha_begin:
                (arch_images, arch_masks), arch_iter = next_batch(arch_iter, arch_loader)
                arch_images = arch_images.to(device, non_blocking=True)
                arch_masks = arch_masks.to(device, non_blocking=True)
                arch_optimizer.zero_grad(set_to_none=True)
                arch_logits = model(arch_images)
                arch_loss = SoftIoULoss(arch_logits, arch_masks)
                arch_loss.backward()
                arch_optimizer.step()
                arch_loss_sum += float(arch_loss.detach().cpu())
                arch_steps += 1

            weight_optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = SoftIoULoss(logits, masks)
            loss.backward()
            clip_grad_norm_(weight_params, args.grad_clip)
            weight_optimizer.step()
            train_loss_sum += float(loss.detach().cpu())

        epoch_sec = time.time() - epoch_start
        train_loss = train_loss_sum / len(weight_loader)
        arch_loss = arch_loss_sum / arch_steps if arch_steps else None
        elapsed_h = (time.time() - start_wall) / 3600.0
        genotype = str(model.genotype())
        if epoch >= args.alpha_begin:
            if genotype == last_genotype:
                stable_epochs += 1
            else:
                stable_epochs = 0
            last_genotype = genotype

        row = {
            "epoch": epoch,
            "epoch_sec": epoch_sec,
            "elapsed_h": elapsed_h,
            "gpu_hours": elapsed_h,
            "train_loss": train_loss,
            "arch_loss": arch_loss,
            "stable_epochs": stable_epochs,
            "genotype": genotype,
        }
        history.append(row)
        log(
            "epoch={}/{} epoch_sec={:.3f} elapsed_h={:.6f} gpu_h={:.6f} train_loss={:.6f} arch_loss={} stable_epochs={}".format(
                epoch + 1,
                args.epochs,
                epoch_sec,
                elapsed_h,
                elapsed_h,
                train_loss,
                "None" if arch_loss is None else f"{arch_loss:.6f}",
                stable_epochs,
            )
        )

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "genotype": str(model.genotype()),
                    "args": vars(args),
                },
                run_dir / "checkpoint_latest.pt",
            )
            with (run_dir / "history.jsonl").open("w") as f:
                for item in history:
                    f.write(json.dumps(item) + "\n")

        completed_epochs = epoch + 1
        if (
            not args.disable_stability_stop
            and epoch + 1 >= args.min_epochs
            and epoch >= args.alpha_begin
            and stable_epochs >= args.patience
        ):
            stop_reason = f"genotype_stable_{args.patience}_epochs"
            log(f"stability_stop epoch={epoch + 1} stable_epochs={stable_epochs}")
            break

    end_dt = datetime.now()
    elapsed_h = (time.time() - start_wall) / 3600.0
    summary = {
        "status": "complete",
        "start": start_dt.isoformat(sep=" ", timespec="seconds"),
        "end": end_dt.isoformat(sep=" ", timespec="seconds"),
        "elapsed_seconds": time.time() - start_wall,
        "elapsed_hours": elapsed_h,
        "gpus": 1,
        "gpu_hours": elapsed_h,
        "completed_epochs": completed_epochs,
        "stop_reason": stop_reason,
        "run_dir": str(run_dir),
        "final_genotype": str(model.genotype()),
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"complete elapsed_h={elapsed_h:.6f} gpu_h={elapsed_h:.6f}")
    log(f"summary={summary_path}")


if __name__ == "__main__":
    main()
