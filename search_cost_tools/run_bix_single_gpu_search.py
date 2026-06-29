#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import lr_scheduler


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU BIX search cost runner")
    parser.add_argument("--repo", default="/home/intern/nas/NAS-BIX-Inference/NAS-BIX-Inference")
    parser.add_argument("--data-root", default="/home/intern/proxylessnas/datasetyhy")
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument("--split-method", default="50_50")
    parser.add_argument("--suffix", default=".png")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--phase1-epochs", type=int, default=100, help="Safety cap, not the Phase1 stopping point")
    parser.add_argument("--phase1-min-epochs", type=int, default=5)
    parser.add_argument("--phase1-patience", type=int, default=5)
    parser.add_argument("--phase2-epochs", type=int, default=100, help="Safety cap per candidate set")
    parser.add_argument("--phase2-min-epochs", type=int, default=5)
    parser.add_argument("--phase2-patience", type=int, default=5)
    parser.add_argument("--phase1-batch-size", type=int, default=16)
    parser.add_argument("--phase1-test-batch-size", type=int, default=16)
    parser.add_argument("--phase2-batch-size", type=int, default=4)
    parser.add_argument("--phase2-test-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--iter", type=int, default=3)
    parser.add_argument("--phase1-multiplier", type=float, default=1.0)
    parser.add_argument("--phase2-multiplier", type=float, default=1.5)
    parser.add_argument("--random-num", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/home/intern/proxylessnas/search_cost_tools/runs/bix_single_gpu_search")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_common_args(args, run_type, epochs, train_batch_size, test_batch_size, multiplier, gene):
    return SimpleNamespace(
        Phase1=(run_type == "all_search_2"),
        epochs=epochs,
        mode="TXT",
        model="BixNAS",
        min_lr=1e-5,
        root=args.data_root,
        lr=0.01,
        lr_decay=3e-5,
        optimizer="Adagrad",
        scheduler="CosineAnnealingLR",
        in_channel=3,
        num_class=1,
        multiplier=multiplier,
        iter=args.iter,
        save_result=False,
        dataset=args.dataset,
        split_method=args.split_method,
        base_size=args.base_size,
        crop_size=args.crop_size,
        suffix=args.suffix,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        workers=args.workers,
        gpu=args.gpu,
        lr_mode="fixed_lr",
        downlayer="five",
        backend="pytorch",
        save_gene=True,
        with_att=False,
        type=run_type,
        backbone="resnet_10",
        random_num=args.random_num,
        conv_type="resblock_more_bone_more_bn",
        gene=gene,
        inference_path="",
        repeated_num=100,
        now=datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
        evaluate_only=False,
    )


def write_phase1_gene(model, gene_path, get_skip):
    encoder, decoder = get_skip(model, "BixNAS")
    level = 5
    with gene_path.open("w") as f:
        f.write("%.3f_%f\n" % (0.0, 0.0))
        for it in range(len(encoder) // level):
            for l in range(level):
                f.write("enc_" + str(it) + "_" + str(l))
                for item in encoder[it * level + l]:
                    f.write("_" + str(item))
                f.write("\n")
            for l in range(level):
                f.write("dec_" + str(it) + "_" + str(l))
                for item in decoder[it * level + l]:
                    f.write("_" + str(item))
                f.write("\n")


def phase1_gene_signature(model, get_skip):
    encoder, decoder = get_skip(model, "BixNAS")
    return json.dumps({"encoder": encoder, "decoder": decoder}, sort_keys=True)


def run_phase1(bix_args, control_args, log):
    from Model.Phase1model_BIX import Phase1_BIX
    from Model.Phase1tools import mIoU, get_skip, load_data
    from Model.dataloaders.loss import SoftIoULoss

    train_set, test_set, train_loader, test_loader, val_img_ids, _ = load_data(bix_args)
    model = Phase1_BIX(
        in_channel=bix_args.in_channel,
        iterations=bix_args.iter,
        num_classes=bix_args.num_class,
        multiplier=bix_args.multiplier,
    ).cuda().float()

    optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=bix_args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=bix_args.epochs, eta_min=bix_args.min_lr)
    best_iou = -1.0
    best_log_path = Path(bix_args.save_path).parent / "best_IoU_IoU.log"
    last_gene = None
    stable_epochs = 0
    stop_reason = "phase1_max_epochs_reached"
    completed_epochs = 0

    for epoch in range(bix_args.epochs):
        epoch_start = time.time()
        model.train()
        train_losses = []
        for data, labels in train_loader:
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = SoftIoULoss(output, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        evaluator = mIoU(1)
        test_losses = []
        model.eval()
        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                output = model(data)
                loss = SoftIoULoss(output, labels)
                test_losses.append(float(loss.detach().cpu()))
                evaluator.update(output, labels)
        _, mean_iou = evaluator.get()
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        test_loss = float(np.mean(test_losses)) if test_losses else 0.0

        if mean_iou > best_iou:
            best_iou = float(mean_iou)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": test_loss,
                    "mean_IOU": best_iou,
                },
                Path(bix_args.save_path).parent / f"mIoU_{bix_args.save_dir}_epoch.pth.tar",
            )
            with best_log_path.open("a") as f:
                f.write(
                    "{} - {:04d}:\t - train_loss: {:04f}:\t - test_loss: {:04f}:\t  - PD: {:04f}:\t - FA: {:04f}:\t mIoU {:.4f}\n".format(
                        datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                        epoch,
                        train_loss,
                        test_loss,
                        0.0,
                        0.0,
                        best_iou,
                    )
                )

        gene_sig = phase1_gene_signature(model, get_skip)
        if gene_sig == last_gene:
            stable_epochs += 1
        else:
            stable_epochs = 0
            last_gene = gene_sig
        completed_epochs = epoch + 1
        log(
            "phase1 epoch={}/{} epoch_sec={:.3f} train_loss={:.6f} test_loss={:.6f} mean_iou={:.6f} stable_epochs={}".format(
                epoch + 1,
                bix_args.epochs,
                time.time() - epoch_start,
                train_loss,
                test_loss,
                float(mean_iou),
                stable_epochs,
            )
        )
        if epoch + 1 >= control_args.phase1_min_epochs and stable_epochs >= control_args.phase1_patience:
            stop_reason = f"phase1_gene_stable_{control_args.phase1_patience}_epochs"
            log(f"phase1 stability_stop epoch={epoch + 1} stable_epochs={stable_epochs}")
            break

    gene_path = Path(bix_args.save_path).parent / "phase1_gene.txt"
    write_phase1_gene(model, gene_path, get_skip)
    log(f"phase1_gene={gene_path}")
    return gene_path, {"completed_epochs": completed_epochs, "stop_reason": stop_reason}


def make_early_phase2_train(control_args, log):
    def train(args, new_networks, iterations, train_loader, test_loader):
        from Model.Phase1tools import get_flops, save_model_phase2
        from Model.Phase2model_BIX import Phase2_BIX
        from Model.dataloaders.loss import SoftIoULoss
        from Model.metrics import iou

        model = Phase2_BIX(
            iterations=args.iter,
            num_classes=args.num_class,
            multiplier=args.multiplier,
            gene=new_networks[0],
        ).cuda().float()

        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == "Adagrad":
            optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        else:
            raise NotImplementedError(args.optimizer)

        if args.scheduler == "CosineAnnealingLR":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
        elif args.scheduler == "StepLR":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        else:
            scheduler = None

        best_ious = [0.0 for _ in range(len(new_networks))]
        stale_epochs = 0
        completed_epochs = 0
        for epoch in range(args.epochs):
            epoch_start = time.time()
            model.train()
            train_losses = []
            for data, labels in train_loader:
                data = data.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                train_total_loss = None
                return_value = model(data, iterations=iterations, profile=False)
                for new_network in new_networks:
                    model.reset_gene(new_network)
                    output = model(data, iterations=iterations, return_value=return_value, profile=False)
                    loss = SoftIoULoss(output, labels)
                    train_total_loss = loss if train_total_loss is None else train_total_loss + loss
                train_total_loss = train_total_loss / len(new_networks)
                train_total_loss.backward()
                optimizer.step()
                train_losses.append(float(train_total_loss.detach().cpu()))
            if scheduler is not None:
                scheduler.step()

            val_ious = [0.0 for _ in range(len(new_networks))]
            model.eval()
            with torch.no_grad():
                for data, labels, _ in test_loader:
                    data = data.cuda(non_blocking=True)
                    labels = labels.cuda(non_blocking=True)
                    return_value = model(data, iterations=iterations, profile=False)
                    for idx, new_network in enumerate(new_networks):
                        model.reset_gene(new_network)
                        output = model(data, iterations=iterations, return_value=return_value, profile=False)
                        output_np, labels_np = output.detach().cpu().numpy(), labels.cpu().numpy()
                        val_ious[idx] += iou(labels_np, output_np, threshold=0)

            improved = False
            for idx, val_iou in enumerate(val_ious):
                score = float(val_iou / len(test_loader))
                if score > best_ious[idx]:
                    best_ious[idx] = score
                    improved = True
                    save_model_phase2(args, model.state_dict(), epoch, float(np.mean(train_losses)), best_ious[idx])

            stale_epochs = 0 if improved else stale_epochs + 1
            completed_epochs = epoch + 1
            log(
                "phase2 iteration={} epoch={}/{} epoch_sec={:.3f} candidates={} train_loss={:.6f} best_iou={:.6f} stale_epochs={}".format(
                    iterations,
                    epoch + 1,
                    args.epochs,
                    time.time() - epoch_start,
                    len(new_networks),
                    float(np.mean(train_losses)) if train_losses else 0.0,
                    max(best_ious) if best_ious else 0.0,
                    stale_epochs,
                )
            )
            if epoch + 1 >= control_args.phase2_min_epochs and stale_epochs >= control_args.phase2_patience:
                log(
                    "phase2 early_stop iteration={} epoch={} stale_epochs={}".format(
                        iterations,
                        epoch + 1,
                        stale_epochs,
                    )
                )
                break

        scores = []
        for idx, new_network in enumerate(new_networks):
            macs, params = get_flops(args, model, new_network)
            scores.append([1.0 - best_ious[idx], macs])
            print(f"model #{idx}: iou {best_ious[idx]}, macs {macs}, params {params}")
        log(f"phase2 iteration={iterations} completed_epochs={completed_epochs}")
        return scores

    return train


def run_phase2(bix_args, control_args, log):
    import Model.Phase1tools as phase1tools
    import Model.Phase2utils as phase2utils

    original_load_data = phase2utils.load_data
    original_train = phase2utils.train
    phase2utils.load_data = lambda a: phase1tools.load_data(a)[:5]
    phase2utils.train = make_early_phase2_train(control_args, log)
    try:
        phase2utils.search(bix_args)
    finally:
        phase2utils.load_data = original_load_data
        phase2utils.train = original_train
    log("phase2 search finished")


def main():
    args = parse_args()
    repo = Path(args.repo).resolve()
    output_root = Path(args.output_dir)
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    summary_path = run_dir / "summary.json"

    def log(message):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU search cost measurement")

    sys.path.insert(0, str(repo))
    os.chdir(repo)
    from Model.dataloaders.utils import save_path

    total_start = time.time()
    log("BIX single-GPU search started")
    log(f"repo={repo}")
    log(f"args={vars(args)}")
    log(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    phase1_args = make_common_args(
        args,
        run_type="all_search_1_search_2",
        epochs=args.phase1_epochs,
        train_batch_size=args.phase1_batch_size,
        test_batch_size=args.phase1_test_batch_size,
        multiplier=args.phase1_multiplier,
        gene=None,
    )
    phase1_args.save_dir, phase1_args.save_path = save_path(phase1_args)
    phase1_start = time.time()
    phase1_gene, phase1_info = run_phase1(phase1_args, args, log)
    phase1_elapsed = time.time() - phase1_start

    phase2_args = make_common_args(
        args,
        run_type="all_search_2",
        epochs=args.phase2_epochs,
        train_batch_size=args.phase2_batch_size,
        test_batch_size=args.phase2_test_batch_size,
        multiplier=args.phase2_multiplier,
        gene=str(phase1_gene),
    )
    phase2_args.save_dir, phase2_args.save_path = save_path(phase2_args)
    phase2_start = time.time()
    run_phase2(phase2_args, args, log)
    phase2_elapsed = time.time() - phase2_start

    total_elapsed = time.time() - total_start
    summary = {
        "status": "complete",
        "start": datetime.fromtimestamp(total_start).isoformat(sep=" ", timespec="seconds"),
        "end": datetime.now().isoformat(sep=" ", timespec="seconds"),
        "elapsed_seconds": total_elapsed,
        "elapsed_hours": total_elapsed / 3600.0,
        "gpus": 1,
        "gpu_hours": total_elapsed / 3600.0,
        "phase1_seconds": phase1_elapsed,
        "phase1_gpu_hours": phase1_elapsed / 3600.0,
        "phase1_completed_epochs": phase1_info["completed_epochs"],
        "phase1_stop_reason": phase1_info["stop_reason"],
        "phase2_seconds": phase2_elapsed,
        "phase2_gpu_hours": phase2_elapsed / 3600.0,
        "phase1_dir": str(Path(phase1_args.save_path).parent),
        "phase2_dir": str(Path(phase2_args.save_path).parent),
        "phase1_gene": str(phase1_gene),
        "run_dir": str(run_dir),
        "args": vars(args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"complete elapsed_h={summary['elapsed_hours']:.6f} gpu_h={summary['gpu_hours']:.6f}")
    log(f"summary={summary_path}")


if __name__ == "__main__":
    main()
