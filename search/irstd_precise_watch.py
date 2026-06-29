#!/usr/bin/env python3
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path("/home/intern/proxylessnas")
LOGS = ROOT / "search" / "logs"
STATUS_PATH = Path("/tmp/irstd_precise_checkpoint_status.txt")
HIT_PATH = Path("/tmp/irstd_precise_checkpoint_hit.txt")
TARGET = 0.6816


def audit():
    rows = []
    for run_dir in sorted(LOGS.glob("*_Retrain*")):
        run_config = run_dir / "run.config"
        checkpoint = run_dir / "checkpoint" / "checkpoint.pth.tar"
        if not run_config.exists() or not checkpoint.exists():
            continue

        try:
            config = json.load(open(run_config))
        except Exception:
            continue
        if config.get("dataset") != "IRSTD-SIRST" or config.get("split_method") != "80_20":
            continue

        try:
            obj = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        except Exception as exc:
            rows.append((None, None, str(run_dir), f"load_error={exc}"))
            continue

        best_iou = obj.get("best_IOU")
        if best_iou is None:
            continue
        rows.append(
            (
                float(best_iou),
                obj.get("epoch"),
                str(run_dir),
                (run_dir / "checkpoint" / "model_best.pth.tar").exists(),
            )
        )
    return rows


def write_status(rows):
    valid_rows = [row for row in rows if row[0] is not None]
    valid_rows.sort(key=lambda row: row[0], reverse=True)
    lines = [
        f"updated: {datetime.now().isoformat(timespec='seconds')}",
        f"target: best_IOU > {TARGET:.4f}",
        f"precision_hits: {sum(1 for value, *_ in valid_rows if value > TARGET)}",
        "",
        "top_checkpoints:",
    ]
    for value, epoch, run_dir, has_best in valid_rows[:30]:
        lines.append(f"{value:.8f} epoch={epoch} best_ckpt={has_best} {run_dir}")
    STATUS_PATH.write_text("\n".join(lines) + "\n")

    hit = next((row for row in valid_rows if row[0] > TARGET), None)
    if hit:
        value, epoch, run_dir, has_best = hit
        HIT_PATH.write_text(
            f"hit at {datetime.now().isoformat(timespec='seconds')}\n"
            f"best_IOU: {value:.8f}\n"
            f"epoch: {epoch}\n"
            f"best_ckpt: {has_best}\n"
            f"run: {run_dir}\n"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    while True:
        rows = audit()
        write_status(rows)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
