#!/usr/bin/env python3
"""Plot search-stage training-loss convergence curves from NAS logs."""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


DARTS_LOG = Path(
    "/home/intern/proxylessnas/search/logs/darts_same_gene_dartsops_1500ep/"
    "0,1,2,3_NUAA-SIRST_Super_all_DARTS_01_06_2026_22_37_47/logs/train_console.txt"
)
HNA_LOG = Path(
    "/home/intern/proxylessnas/search/logs/"
    "4,5,6,7_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_08_06_2026_00_54_32/"
    "logs/train_console.txt"
)
PROXYLESS_LOG = Path(
    "/home/intern/nas/proxylessnas-master-SIRST-new-final_share/search/logs/"
    "0,1_NUAA-SIRST_Super_all_Proxyless_01_12_2023_10_33_21/logs/train_console.txt"
)


TRAIN_LOSS_RE = re.compile(
    r"^(Warmup\s+)?Train\s+\[(\d+)\]\[(\d+)/(\d+)\].*?"
    r"\bLoss\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)"
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    path: Path
    color: str


RUNS = [
    RunSpec("HNA-NAS", HNA_LOG, "#c00000"),
    RunSpec("ProxylessNAS", PROXYLESS_LOG, "#38aeea"),
    RunSpec("DARTS", DARTS_LOG, "#2fa84f"),
]


def parse_train_loss(path: Path, phase: str, max_epochs: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Return one averaged training-loss value per epoch.

    phase:
      - search: only non-Warmup Train lines
      - warmup: only Warmup Train lines
      - all: Warmup Train followed by Train
    """

    if not path.is_file():
        raise FileNotFoundError(path)

    # key -> (first_seen_order, largest_logged_batch_idx, epoch_avg_loss)
    rows: Dict[Tuple[str, int], Tuple[int, int, float]] = {}
    next_order = 0

    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_LOSS_RE.search(line)
        if not match:
            continue

        is_warmup = bool(match.group(1))
        row_phase = "warmup" if is_warmup else "search"
        if phase != "all" and row_phase != phase:
            continue

        epoch = int(match.group(2))
        batch_idx = int(match.group(3))
        avg_loss = float(match.group(6))
        key = (row_phase, epoch)

        if key not in rows:
            rows[key] = (next_order, batch_idx, avg_loss)
            next_order += 1
            continue

        order, old_batch_idx, old_loss = rows[key]
        if batch_idx >= old_batch_idx:
            rows[key] = (order, batch_idx, avg_loss)
        else:
            rows[key] = (order, old_batch_idx, old_loss)

    if not rows:
        raise ValueError(f"No {phase!r} training-loss records found in {path}")

    ordered_rows = sorted(rows.values(), key=lambda item: item[0])
    if max_epochs > 0:
        ordered_rows = ordered_rows[:max_epochs]
    x = np.arange(1, len(ordered_rows) + 1, dtype=float)
    y = np.array([item[2] for item in ordered_rows], dtype=float)
    return x, y


def smooth_curve(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) <= 2:
        return y.copy()

    window = min(window, len(y))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return y.copy()

    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(padded, kernel, mode="valid")


def normalize_x(x: np.ndarray, x_max: float) -> np.ndarray:
    if len(x) == 1:
        return np.array([0.0], dtype=float)
    return np.linspace(0.0, x_max, len(x), dtype=float)


def plot_curves(args: argparse.Namespace) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.unicode_minus": False,
        }
    )

    series = []
    for run in RUNS:
        raw_x, raw_y = parse_train_loss(run.path, phase=args.phase, max_epochs=args.max_epochs)
        y = smooth_curve(raw_y, args.smooth_window)
        x = normalize_x(raw_x, args.x_max) if args.normalize_x else raw_x
        series.append((run, x, y, raw_y))

    y_min = min(float(np.min(y)) for _, _, y, _ in series)
    y_max = max(float(np.max(y)) for _, _, y, _ in series)
    y_span = max(y_max - y_min, 1e-6)
    axis_y = y_min - 0.09 * y_span
    y_top = y_max + 0.10 * y_span
    x_right = args.x_max if args.normalize_x else max(float(x[-1]) for _, x, _, _ in series)
    x_axis_right = x_right * 1.14

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    for run, x, y, _ in series:
        ax.plot(x, y, color=run.color, lw=3.0, solid_capstyle="round", label=run.label)

    final_levels = [float(y[-1]) for _, _, y, _ in series]
    for level in final_levels:
        ax.hlines(level, 0, x_right, colors="black", linestyles=(0, (8, 5)), lw=1.0)
        ax.text(
            -0.18 if args.normalize_x else -0.018 * x_right,
            level,
            f"{level:.2f}",
            ha="right",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
    ax.vlines(
        x_right,
        min(final_levels),
        max(final_levels),
        colors="black",
        linestyles=(0, (8, 5)),
        lw=1.0,
    )

    ax.set_xlim(-0.11 * x_right, x_axis_right)
    ax.set_ylim(axis_y - 0.03 * y_span, y_top)
    ax.axis("off")

    arrow_width = 0.0015 * y_span
    head_width = 0.035 * y_span
    head_length = 0.035 * x_right
    ax.arrow(
        0,
        axis_y,
        x_axis_right,
        0,
        head_width=head_width,
        head_length=head_length,
        fc="black",
        ec="black",
        lw=1.2,
        length_includes_head=True,
        width=arrow_width,
    )
    ax.arrow(
        0,
        axis_y,
        0,
        y_top - axis_y,
        head_width=0.018 * x_right,
        head_length=0.055 * y_span,
        fc="black",
        ec="black",
        lw=1.2,
        length_includes_head=True,
        width=0.0007 * x_right,
    )

    ax.text(
        -0.095 * x_right,
        axis_y + 0.58 * (y_top - axis_y),
        "Training Loss\n(Search Stage)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
    )
    ax.text(
        x_right,
        axis_y - 0.045 * y_span,
        f"{args.x_max:g}" if args.normalize_x else f"{int(x_right)}",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        x_axis_right,
        axis_y - 0.045 * y_span,
        "Epoch",
        ha="right",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    handles = [
        Rectangle((0, 0), 1, 1, facecolor=run.color, edgecolor="black")
        for run, _, _, _ in series
    ]
    labels = [run.label for run, _, _, _ in series]
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.88, 0.93),
        prop={"family": "serif", "weight": "bold", "size": 13},
        handlelength=1.2,
        handleheight=1.2,
    )

    if args.caption:
        fig.text(
            0.08,
            0.02,
            args.caption,
            ha="left",
            va="bottom",
            fontsize=13,
            color="blue",
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    if args.pdf:
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)

    for run, _, y, raw_y in series:
        print(
            f"{run.label}: records={len(raw_y)}, "
            f"raw_start={raw_y[0]:.4f}, raw_end={raw_y[-1]:.4f}, "
            f"smooth_end={y[-1]:.4f}"
        )
    print(f"saved: {output}")
    if args.pdf:
        print(f"saved: {output.with_suffix('.pdf')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot HNA-NAS, ProxylessNAS and DARTS search-loss convergence curves."
    )
    parser.add_argument(
        "--output",
        default="/home/intern/proxylessnas/search/figures/search_loss_convergence_real_logs.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--phase",
        choices=("search", "warmup", "all"),
        default="search",
        help="Which training phase to plot. Default skips Warmup lines.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd moving-average window in epochs. Use 1 to disable smoothing.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Use only the first N epoch records after phase filtering. Use 0 for all.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=10.0,
        help="Right edge of normalized x-axis.",
    )
    parser.add_argument(
        "--raw-epoch",
        action="store_false",
        dest="normalize_x",
        help="Use raw epoch indices instead of normalizing each run to [0, x-max].",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF copy.")
    parser.add_argument("--show", action="store_true", help="Show the plot window.")
    parser.add_argument(
        "--caption",
        default="",
        help="Optional caption drawn under the figure.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_curves(args)


if __name__ == "__main__":
    main()
