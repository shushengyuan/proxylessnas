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


def parse_args():
    parser = argparse.ArgumentParser(description="Single-GPU ProxylessNAS search cost runner")
    parser.add_argument("--repo", default="/home/intern/nas/proxylessnas-master-SIRST-new-final_share")
    parser.add_argument("--data-root", default="/home/intern/proxylessnas/datasetyhy")
    parser.add_argument(
        "--gene",
        default="/home/intern/proxylessnas/search1yhy/0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new/phase1_gene.txt",
    )
    parser.add_argument("--dataset", default="NUAA-SIRST")
    parser.add_argument("--split-method", default="50_50")
    parser.add_argument("--suffix", default=".png")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--warmup-epochs", type=int, default=300)
    parser.add_argument("--search-epochs", type=int, default=500, help="Safety cap, not the search stopping point")
    parser.add_argument("--min-search-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=20, help="Stop after this many unchanged discrete architectures")
    parser.add_argument("--disable-stability-stop", action="store_true")
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--test-batch-size", type=int, default=8)
    parser.add_argument("--valid-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="/home/intern/proxylessnas/search_cost_tools/runs/proxyless_single_gpu_search",
    )
    return parser.parse_args()


def setup_imports(repo):
    repo = Path(repo).resolve()
    search_dir = repo / "search"
    sys.path.insert(0, str(repo))
    sys.path.insert(1, str(search_dir))
    return repo


def seed_everything(seed, torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_proxyless_args(control_args, proxyless_log_root):
    return SimpleNamespace(
        path=str(proxyless_log_root),
        resume=False,
        manual_seed=control_args.seed,
        init_lr=0.01,
        lr_schedule_type="cosine",
        dataset=control_args.dataset,
        train_batch_size=control_args.train_batch_size,
        test_batch_size=control_args.test_batch_size,
        valid_size=control_args.valid_size,
        opt_type="Adagrad",
        momentum=0.9,
        no_nesterov=False,
        weight_decay=4e-5,
        label_smoothing=0.1,
        no_decay_keys=None,
        model_init="he_fout",
        init_div_groups=False,
        validation_frequency=1,
        print_frequency=10,
        n_worker=control_args.workers,
        resize_scale=0.08,
        distort_color="normal",
        id_mode="TXT",
        root=control_args.data_root,
        split_method=control_args.split_method,
        base_size=control_args.base_size,
        crop_size=control_args.crop_size,
        suffix=control_args.suffix,
        eval_batch_size=1,
        num_class=1,
        iterations=5,
        conv_type="res_add_new_new",
        channel_num="two",
        in_channel=3,
        arch_algo="grad",
        arch_init_type="normal",
        arch_init_ratio=0.001,
        arch_opt_type="adam",
        arch_lr=0.005,
        arch_adam_beta1=0.0,
        arch_adam_beta2=0.999,
        arch_adam_eps=1e-8,
        arch_weight_decay=0,
        target_hardware="flops",
        fast="False",
        grad_update_arch_param_every=5,
        grad_update_steps=1,
        grad_binary_mode="full_v2",
        grad_data_batch=None,
        grad_reg_loss_type="add#linear",
        grad_reg_loss_lambda=0.1,
        grad_reg_loss_alpha=0.2,
        grad_reg_loss_beta=0.3,
        ref_value=7 * 1e9,
        random_choose="False",
        warmup_epochs=control_args.warmup_epochs,
        n_epochs=control_args.search_epochs,
        retrain_epoch=1500,
        retrain_init_lr=0.05,
        retrain_fixed_lr=0.01,
        retrain_lr_schedule_type="fixed",
        retrain_valid_size=1,
        retrain_latency="gpu",
        gpu=control_args.gpu,
        retrain_log_path="0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_26_10_2023_12_47_26",
        add_decoder="False",
        add_encoder0="True",
        mode="all_search_train",
        model_name="Super_all",
        candidates_type="Proxyless",
        gene=control_args.gene,
        ROC_thr=10,
        postprocess="none",
        Inference_resize="True",
        Inference_repeated=1,
        backbone="resnet_18",
        more_down="False",
    )


def build_manager(args):
    from models import SIRSTRunConfig
    from models.super_nets.super_proxyless_SIRST import SuperProxylessNASNets
    from nas_manager import ArchSearchRunManager, GradientArchSearchConfig
    from search.utils.utils import (
        conv_name_define,
        decode_gene,
        operation_choose,
        random_operation_choose,
        save_path,
        save_train_log,
    )

    ref_values = {
        "flops": {
            "16.00": 59 * 1e6,
            "32.00": 97 * 1e6,
            "48.00": 209 * 1e6,
            "two": 0.9 * 1e9,
        },
        "params": {"16.00": 1 * 1e6, "32.00": 2 * 1e6, "48.00": 3 * 1e6, "two": 2.5 * 1e6},
        "cpu": {"16.00": 10, "32.00": 20, "two": 90},
        "gpu": {"16.00": 5, "32.00": 10, "two": 4},
        "edge-cpu": {"16.00": 10, "32.00": 20, "two": 1200},
        "edge-gpu": {"16.00": 5, "32.00": 10, "two": 100},
    }

    gene_param_path = args.gene.split("phase1_gene")[0] + "train_parameters.txt"
    args.backbone, args.channel_num, args.iterations, args.conv_type, args.add_decoder, args.add_encoder0, args.more_down = decode_gene(
        gene_param_path
    )

    save_dir = save_path(args.gpu, args.dataset, args.model_name, args.candidates_type)
    args.path = str(Path(args.path) / save_dir)
    os.makedirs(args.path, exist_ok=True)
    save_train_log(args, args.path)

    args.lr_schedule_param = None
    args.opt_param = {"momentum": args.momentum, "nesterov": not args.no_nesterov}
    run_config = SIRSTRunConfig(**args.__dict__)

    if args.random_choose == "True":
        args.conv_candidates = random_operation_choose(args.candidates_type, args.iterations, args.random_choose)
    else:
        args.conv_candidates = operation_choose(args.candidates_type)
    args.conv_name = conv_name_define(args.conv_candidates, args.iterations)

    super_net = SuperProxylessNASNets(
        conv_type=args.conv_type,
        iterations=args.iterations,
        backbone=args.backbone,
        in_channel=args.in_channel,
        conv_candidates=args.conv_candidates,
        n_classes=args.num_class,
        channel_num=args.channel_num,
        model_name=args.model_name,
        gene=args.gene,
        add_decoder=args.add_decoder,
        add_encoder0=args.add_encoder0,
        more_down=args.more_down,
    )

    if args.arch_opt_type == "adam":
        args.arch_opt_param = {"betas": (args.arch_adam_beta1, args.arch_adam_beta2), "eps": args.arch_adam_eps}
    else:
        args.arch_opt_param = None

    if args.target_hardware is None:
        args.ref_value = None
    elif args.grad_reg_loss_type != "add#linear_abs":
        args.ref_value = ref_values[args.target_hardware][args.channel_num]

    if args.grad_reg_loss_type == "add#linear":
        args.grad_reg_loss_params = {"lambda": args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == "mul#log":
        args.grad_reg_loss_params = {"alpha": args.grad_reg_loss_alpha, "beta": args.grad_reg_loss_beta}
    else:
        args.grad_reg_loss_params = None

    arch_search_config = GradientArchSearchConfig(**args.__dict__)
    manager = ArchSearchRunManager(
        args.path,
        super_net,
        args.crop_size,
        run_config,
        arch_search_config,
        args.conv_name,
        args.target_hardware,
        args.fast,
        args.model_name,
        args.iterations,
        args.mode,
        args.gene,
        args.candidates_type,
        args.random_choose,
    )
    return manager, args


def architecture_signature(manager):
    signature = []
    for idx, block in enumerate(manager.net.encoders):
        op = getattr(block, "operator", None)
        if op is None:
            signature.append([idx, "none"])
            continue
        if getattr(block, "skipped", False):
            signature.append([idx, "skipped"])
            continue
        if hasattr(op, "chosen_index"):
            chosen_idx, _ = op.chosen_index
            chosen_op = op.candidate_ops[chosen_idx]
            op_name = chosen_op.module_str() if chosen_op is not None else "none"
            signature.append([idx, int(chosen_idx), op_name])
        else:
            signature.append([idx, str(op)])
    return signature


def train_with_stability_stop(manager, control_args, total_start, history_path, log):
    import torch
    from search.utils.utils import SoftIoULoss, mIoU, primary_output
    from utils import AverageMeter, count_parameters

    manager.mIoU = mIoU(1)
    data_loader = manager.run_manager.run_config.train_loader
    n_batch = len(data_loader)

    arch_param_num = len(list(manager.net.architecture_parameters()))
    binary_gates_num = len(list(manager.net.binary_gates()))
    weight_param_num = len(list(manager.net.weight_parameters()))
    log(f"arch_params={arch_param_num} binary_gates={binary_gates_num} weight_params={weight_param_num}")
    update_schedule = manager.arch_search_config.get_update_schedule(n_batch)

    last_signature = None
    stable_epochs = 0
    stop_reason = "max_epochs_reached"
    completed_epochs = 0
    final_signature = None

    for epoch in range(manager.run_manager.start_epoch, manager.run_manager.run_config.n_epochs):
        epoch_start = time.time()
        manager.mIoU.reset()
        print("\n", "-" * 30, "Train epoch: %d" % (epoch + 1), "-" * 30, "\n")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        entropy = AverageMeter()
        manager.run_manager.net.train()
        end = time.time()
        train_iou = 0.0
        lr = manager.run_manager.run_config.init_lr

        for i, (images, labels) in enumerate(data_loader):
            data_time.update(time.time() - end)
            lr = manager.run_manager.run_config.adjust_learning_rate(
                manager.run_manager.optimizer, epoch, batch=i, nBatch=n_batch
            )

            net_entropy = manager.net.entropy()
            entropy.update(net_entropy.data.item() / arch_param_num, 1)

            images, labels = images.to(manager.run_manager.device), labels.to(manager.run_manager.device)
            manager.net.reset_binary_gates()
            manager.net.unused_modules_off()
            output = primary_output(manager.run_manager.net(images))

            loss = SoftIoULoss(output, labels)
            losses.update(loss, images.size(0))
            manager.mIoU.update(output, labels)
            _, train_iou = manager.mIoU.get()

            manager.run_manager.net.zero_grad()
            loss.backward()
            manager.run_manager.optimizer.step()
            manager.net.unused_modules_back()

            if epoch > 0:
                for _ in range(update_schedule.get(i, 0)):
                    start_time = time.time()
                    arch_loss, exp_value = manager.gradient_step()
                    used_time = time.time() - start_time
                    log_str = "Architecture [%d-%d]\t Time %.4f\t Loss %.4f\t %s %s" % (
                        epoch + 1,
                        i,
                        used_time,
                        arch_loss,
                        manager.arch_search_config.target_hardware,
                        exp_value,
                    )
                    manager.write_log(log_str, prefix="gradient", should_print=False)

            batch_time.update(time.time() - end)
            end = time.time()

            if i % manager.run_manager.run_config.print_frequency == 0 or i + 1 == n_batch:
                batch_log = (
                    "Train [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    "Entropy {entropy.val:.5f} ({entropy.avg:.5f})\t"
                    "train_IoU {train_iou:.3f}\t"
                    "lr {lr:.5f}\t"
                ).format(
                    epoch + 1,
                    i,
                    n_batch - 1,
                    batch_time=batch_time,
                    data_time=data_time,
                    losses=losses,
                    entropy=entropy,
                    train_iou=train_iou,
                    lr=lr,
                )
                manager.run_manager.write_log(batch_log, "train")

        manager.write_log("-" * 30 + "Current Architecture [%d]" % (epoch + 1) + "-" * 30, prefix="arch")
        for idx, block in enumerate(manager.net.encoders):
            manager.write_log("%d. %s" % (idx, block.operator.module_str), prefix="arch")
        manager.write_log("-" * 60, prefix="arch")

        val_loss = None
        val_mean_iou = None
        if (epoch + 1) % manager.run_manager.run_config.validation_frequency == 0:
            (
                (val_loss, val_mean_iou),
                flops,
                total_params,
                latency,
                theoretical_latency,
                theoretical_flops,
                theoretical_params,
            ) = manager.validate(fast="True")
            manager.run_manager.best_IOU = max(manager.run_manager.best_IOU, val_mean_iou)
            val_log = (
                "Valid [{0}/{1}]\tloss {2:.3f}\tval_mean_IOU {3:.3f} ({4:.3f})\t"
                "Train IoU {train_iou:.3f}\t"
                "Entropy {entropy.val:.5f}\t"
                "Latency-{5}: {6:.3f}ms T-{5}:{9:.3f}ms\t"
                "Flops: {7:.2f}M\t"
                "T-Flops: {10:.2f}M\t"
                "Params: {8:.2f}M\t"
                "T-Params: {11:.2f}M\t"
            ).format(
                epoch + 1,
                manager.run_manager.run_config.n_epochs,
                val_loss,
                val_mean_iou,
                manager.run_manager.best_IOU,
                manager.arch_search_config.target_hardware,
                latency,
                flops / 1e6,
                total_params / 1e6,
                theoretical_latency,
                theoretical_flops / 1e6,
                theoretical_params / 1e6,
                entropy=entropy,
                train_iou=train_iou,
            )
            manager.run_manager.write_log(val_log, "valid")

        manager.run_manager.save_model(
            {
                "warmup": False,
                "epoch": epoch,
                "weight_optimizer": manager.run_manager.optimizer.state_dict(),
                "arch_optimizer": manager.arch_optimizer.state_dict(),
                "state_dict": manager.net.state_dict(),
            }
        )

        final_signature = architecture_signature(manager)
        if final_signature == last_signature:
            stable_epochs += 1
        else:
            stable_epochs = 0
            last_signature = final_signature
        completed_epochs = epoch + 1

        epoch_sec = time.time() - epoch_start
        elapsed_h = (time.time() - total_start) / 3600.0
        row = {
            "epoch": epoch + 1,
            "epoch_sec": epoch_sec,
            "elapsed_h": elapsed_h,
            "gpu_hours": elapsed_h,
            "train_loss": float(losses.avg.item() if hasattr(losses.avg, "item") else losses.avg),
            "train_iou": float(train_iou),
            "valid_loss": None if val_loss is None else float(val_loss),
            "valid_mean_iou": None if val_mean_iou is None else float(val_mean_iou),
            "stable_epochs": stable_epochs,
            "signature": final_signature,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        log(
            "search epoch={}/{} epoch_sec={:.3f} elapsed_h={:.6f} gpu_h={:.6f} train_loss={:.6f} val_iou={} stable_epochs={}".format(
                epoch + 1,
                manager.run_manager.run_config.n_epochs,
                epoch_sec,
                elapsed_h,
                elapsed_h,
                row["train_loss"],
                "None" if val_mean_iou is None else f"{float(val_mean_iou):.6f}",
                stable_epochs,
            )
        )

        if (
            not control_args.disable_stability_stop
            and epoch + 1 >= control_args.min_search_epochs
            and stable_epochs >= control_args.patience
        ):
            stop_reason = f"arch_stable_{control_args.patience}_epochs"
            log(f"stability_stop search_epoch={epoch + 1} stable_epochs={stable_epochs}")
            break

    normal_net = manager.net.cpu().convert_to_normal_net()
    print("Total training params: %.2fM" % (count_parameters(normal_net) / 1e6))
    learned_net_dir = Path(manager.run_manager.path) / "learned_net"
    learned_net_dir.mkdir(parents=True, exist_ok=True)
    (learned_net_dir / "net.config").write_text(json.dumps(normal_net.config(), indent=4), encoding="utf-8")
    (learned_net_dir / "run.config").write_text(
        json.dumps(manager.run_manager.run_config.config, indent=4), encoding="utf-8"
    )
    torch.save(
        {"state_dict": normal_net.state_dict(), "dataset": manager.run_manager.run_config.dataset},
        learned_net_dir / "init",
    )

    return {
        "completed_search_epochs": completed_epochs,
        "stop_reason": stop_reason,
        "final_signature": final_signature,
    }


def main():
    control_args = parse_args()
    start_wall = time.time()
    start_dt = datetime.now()
    os.environ["CUDA_VISIBLE_DEVICES"] = control_args.gpu

    import torch

    repo = setup_imports(control_args.repo)
    seed_everything(control_args.seed, torch)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU search cost measurement")

    output_root = Path(control_args.output_dir)
    run_name = start_dt.strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "run.log"
    history_path = run_dir / "history.jsonl"
    summary_path = run_dir / "summary.json"
    proxyless_log_root = run_dir / "proxyless_logs"
    proxyless_log_root.mkdir(parents=True, exist_ok=True)

    def log(message):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log("ProxylessNAS single-GPU search started")
    log(f"repo={repo}")
    log(f"args={vars(control_args)}")
    log(f"cuda_visible_devices={control_args.gpu}")
    log(f"torch={torch.__version__} device={torch.cuda.get_device_name(0)}")

    setup_start = time.time()
    proxyless_args = make_proxyless_args(control_args, proxyless_log_root)
    manager, proxyless_args = build_manager(proxyless_args)
    setup_seconds = time.time() - setup_start
    log(f"proxyless_log={proxyless_args.path}")
    log(f"setup_seconds={setup_seconds:.3f}")

    warmup_seconds = 0.0
    if manager.warmup and control_args.warmup_epochs > 0:
        warmup_start = time.time()
        manager.warm_up(warmup_epochs=control_args.warmup_epochs, crop_size=control_args.crop_size, lr=proxyless_args.init_lr)
        warmup_seconds = time.time() - warmup_start
        log(f"warmup_complete epochs={control_args.warmup_epochs} seconds={warmup_seconds:.3f}")

    search_start = time.time()
    train_summary = train_with_stability_stop(manager, control_args, start_wall, history_path, log)
    search_seconds = time.time() - search_start

    end_dt = datetime.now()
    elapsed_seconds = time.time() - start_wall
    elapsed_h = elapsed_seconds / 3600.0
    summary = {
        "status": "complete",
        "start": start_dt.isoformat(sep=" ", timespec="seconds"),
        "end": end_dt.isoformat(sep=" ", timespec="seconds"),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_hours": elapsed_h,
        "gpus": 1,
        "gpu_hours": elapsed_h,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "warmup_gpu_hours": warmup_seconds / 3600.0,
        "search_seconds": search_seconds,
        "search_gpu_hours": search_seconds / 3600.0,
        "completed_warmup_epochs": control_args.warmup_epochs,
        "completed_search_epochs": train_summary["completed_search_epochs"],
        "stop_reason": train_summary["stop_reason"],
        "final_signature": train_summary["final_signature"],
        "run_dir": str(run_dir),
        "proxyless_log": proxyless_args.path,
        "history": str(history_path),
        "control_args": vars(control_args),
        "proxyless_args": vars(proxyless_args),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"complete elapsed_h={elapsed_h:.6f} gpu_h={elapsed_h:.6f} stop_reason={train_summary['stop_reason']}")
    log(f"summary={summary_path}")


if __name__ == "__main__":
    main()
