#!/usr/bin/env python3
"""Plot candidate-group IoU and search cost with 200-epoch calibration."""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


VALID_PATTERN = re.compile(
    r"^(?:==========)?(Warmup\s+)?Valid\s+\[(\d+)/(\d+)\].*?"
    r"(?:Validate_IoU|val_mean_IOU|validate_IoU)\s+([0-9.]+)\s*\(([0-9.]+)\)",
    re.IGNORECASE,
)

TRAIN_TIME_PATTERN = re.compile(
    r"^(Warmup\s+)?Train\s+\[(\d+)\]\[(\d+)/(\d+)\]\s+"
    r"Time\s+[0-9.]+\s+\(([0-9.]+)\)"
)


@dataclass(frozen=True)
class PointSpec:
    group: str
    label: str
    ops: int
    effect_log: str
    effect_phase: str
    cost_log: str
    gpu_count: int
    note: str


@dataclass
class Point:
    group: str
    label: str
    ops: int
    effect_source: Path
    effect_phase: str
    effect_200_iou: float
    mapped_iou: float
    search_cost_source: Path
    total_logged_epochs: int
    warmup_epochs_detected: int
    search_epochs_detected: int
    total_wall_hours: float
    search_only_wall_hours: float
    gpu_count: int
    total_gpu_hours: float
    search_only_gpu_hours: float
    note: str


POINT_SPECS = [
    PointSpec(
        group="Res_Group_Spa_MBConv",
        label="Res+Group+Spa+MBConv",
        ops=12,
        effect_log="0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/logs/valid_console.txt",
        effect_phase="train",
        cost_log="0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/logs/train_console.txt",
        gpu_count=2,
        note="200-epoch metric from train-phase validation in the 2023 search log.",
    ),
    PointSpec(
        group="whole",
        label="Whole / full",
        ops=18,
        effect_log="ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/logs/valid_console.txt",
        effect_phase="train",
        cost_log="0,1_NUAA-SIRST_Super_all_whole_all_27_04_2026_00_04_38/logs/train_console.txt",
        gpu_count=2,
        note="IoU anchor from current best full candidate; search cost from the best complete whole_all GPU search log.",
    ),
]


def parse_best_iou(
    valid_log: Path,
    *,
    max_epoch: Optional[int] = None,
    phase: str = "train",
) -> Tuple[Optional[float], int, Optional[int]]:
    if not valid_log.is_file():
        return None, 0, None

    best_iou: Optional[float] = None
    best_epoch: Optional[int] = None
    n_records = 0
    for line in valid_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = VALID_PATTERN.search(line)
        if not match:
            continue
        row_phase = "warmup" if match.group(1) else "train"
        epoch = int(match.group(2))
        if phase != "all" and row_phase != phase:
            continue
        if max_epoch is not None and epoch > max_epoch:
            continue

        best_value = float(match.group(5))
        n_records += 1
        if best_iou is None or best_value > best_iou:
            best_iou = best_value
            best_epoch = epoch
    return best_iou, n_records, best_epoch


def parse_logged_train_cost(train_log: Path) -> Tuple[float, float, int, int]:
    if not train_log.is_file():
        raise FileNotFoundError(train_log)

    final_epoch_rows: Dict[Tuple[str, int], Tuple[int, int, float]] = {}
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_TIME_PATTERN.search(line)
        if not match:
            continue
        phase = "warmup" if match.group(1) else "search"
        epoch = int(match.group(2))
        batch_idx = int(match.group(3))
        last_batch_idx = int(match.group(4))
        avg_batch_seconds = float(match.group(5))
        key = (phase, epoch)
        old = final_epoch_rows.get(key)
        if old is None or batch_idx >= old[0]:
            final_epoch_rows[key] = (batch_idx, last_batch_idx, avg_batch_seconds)

    if not final_epoch_rows:
        raise ValueError(f"no train time rows found: {train_log}")

    total_seconds = 0.0
    search_seconds = 0.0
    warmup_epochs = 0
    search_epochs = 0
    for (phase, _), (_, last_batch_idx, avg_batch_seconds) in final_epoch_rows.items():
        seconds = avg_batch_seconds * (last_batch_idx + 1)
        total_seconds += seconds
        if phase == "warmup":
            warmup_epochs += 1
        else:
            search_epochs += 1
            search_seconds += seconds

    return total_seconds / 3600.0, search_seconds / 3600.0, warmup_epochs, search_epochs


def build_points(logs_root: Path, use_total_search_cost: bool) -> Tuple[List[Point], float, float, float]:
    anchor_log = logs_root / "ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/logs/valid_console.txt"
    anchor_200, n_anchor_200, _ = parse_best_iou(anchor_log, max_epoch=200, phase="train")
    anchor_final, n_anchor_final, _ = parse_best_iou(anchor_log, max_epoch=None, phase="train")
    if anchor_200 is None or anchor_final is None:
        raise RuntimeError(f"failed to parse anchor IoU from {anchor_log}")
    if n_anchor_200 < 200:
        raise RuntimeError(f"anchor has only {n_anchor_200} records within 200 epochs")

    scale = anchor_final / anchor_200
    points: List[Point] = []
    for spec in POINT_SPECS:
        effect_log = logs_root / spec.effect_log
        effect_200, n_effect, _ = parse_best_iou(effect_log, max_epoch=200, phase=spec.effect_phase)
        if effect_200 is None:
            raise RuntimeError(f"failed to parse 200-epoch IoU for {spec.group}: {effect_log}")
        if n_effect < 200:
            print(f"warning: {spec.group} has only {n_effect} effect records within 200 epochs")

        cost_log = logs_root / spec.cost_log
        total_wall_hours, search_wall_hours, warmup_epochs, search_epochs = parse_logged_train_cost(cost_log)
        total_gpu_hours = total_wall_hours * spec.gpu_count
        search_gpu_hours = search_wall_hours * spec.gpu_count
        mapped_iou = effect_200 * scale

        points.append(
            Point(
                group=spec.group,
                label=spec.label,
                ops=spec.ops,
                effect_source=effect_log.resolve(),
                effect_phase=spec.effect_phase,
                effect_200_iou=effect_200,
                mapped_iou=mapped_iou,
                search_cost_source=cost_log.resolve(),
                total_logged_epochs=warmup_epochs + search_epochs,
                warmup_epochs_detected=warmup_epochs,
                search_epochs_detected=search_epochs,
                total_wall_hours=total_wall_hours,
                search_only_wall_hours=search_wall_hours,
                gpu_count=spec.gpu_count,
                total_gpu_hours=total_gpu_hours,
                search_only_gpu_hours=search_gpu_hours,
                note=spec.note,
            )
        )
    return points, anchor_200, anchor_final, scale


def write_csv(path: Path, points: List[Point]) -> None:
    rows = []
    for p in points:
        rows.append(
            {
                "candidate_group": p.group,
                "label": p.label,
                "ops": p.ops,
                "effect_200_iou": f"{p.effect_200_iou:.6f}",
                "mapped_iou": f"{p.mapped_iou:.6f}",
                "effect_phase": p.effect_phase,
                "effect_source": str(p.effect_source),
                "total_logged_epochs": p.total_logged_epochs,
                "warmup_epochs_detected": p.warmup_epochs_detected,
                "search_epochs_detected": p.search_epochs_detected,
                "total_wall_hours": f"{p.total_wall_hours:.6f}",
                "search_only_wall_hours": f"{p.search_only_wall_hours:.6f}",
                "gpu_count": p.gpu_count,
                "total_gpu_hours": f"{p.total_gpu_hours:.6f}",
                "search_only_gpu_hours": f"{p.search_only_gpu_hours:.6f}",
                "search_cost_source": str(p.search_cost_source),
                "note": p.note,
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(points: List[Point], output_png: Path, output_pdf: Path, use_total_search_cost: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(points)))
    labels = [f"{p.label}\n({p.ops} ops)" for p in points]
    ious = [p.mapped_iou for p in points]
    costs = [p.total_gpu_hours if use_total_search_cost else p.search_only_gpu_hours for p in points]
    cost_label = "Search Cost (GPU-hours, warmup+search)" if use_total_search_cost else "Search Cost (GPU-hours, search-only)"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax_iou = plt.subplots(figsize=(9.4, 5.2), dpi=220)
    ax_cost = ax_iou.twinx()

    iou_color = "#165a8f"
    cost_color = "#c86913"
    ax_iou.plot(xs, ious, marker="o", markersize=8, linewidth=2.5, color=iou_color, label="Mapped IoU")
    ax_cost.plot(
        xs,
        costs,
        marker="s",
        markersize=7,
        linewidth=2.3,
        linestyle="--",
        color=cost_color,
        label="Search cost",
    )

    ax_iou.set_xticks(xs)
    ax_iou.set_xticklabels(labels)
    ax_iou.set_xlabel("Search Candidate Group")
    ax_iou.set_ylabel("Mapped IoU", color=iou_color)
    ax_cost.set_ylabel(cost_label, color=cost_color)
    ax_iou.tick_params(axis="y", labelcolor=iou_color)
    ax_cost.tick_params(axis="y", labelcolor=cost_color)
    ax_iou.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.65)
    ax_iou.set_axisbelow(True)

    y_min = max(0.0, min(ious) - 0.04)
    y_max = min(1.0, max(ious) + 0.04)
    if math.isclose(y_min, y_max):
        y_min = max(0.0, y_min - 0.05)
        y_max = min(1.0, y_max + 0.05)
    ax_iou.set_ylim(y_min, y_max)
    ax_cost.set_ylim(0.0, max(costs) * 1.25 if costs else 1.0)
    ax_iou.set_title("Candidate Group vs Calibrated IoU and Search Cost")

    for x, y, point in zip(xs, ious, points):
        ax_iou.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=iou_color,
            fontsize=10,
            fontweight="bold",
        )
        ax_iou.annotate(
            f"200ep {point.effect_200_iou:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 24),
            ha="center",
            color=iou_color,
            fontsize=8,
        )
    for x, y in zip(xs, costs):
        ax_cost.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -17),
            ha="center",
            color=cost_color,
            fontsize=10,
            fontweight="bold",
        )

    handles_iou, labels_iou = ax_iou.get_legend_handles_labels()
    handles_cost, labels_cost = ax_cost.get_legend_handles_labels()
    ax_iou.legend(handles_iou + handles_cost, labels_iou + labels_cost, loc="upper center", ncol=2, frameon=False)

    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    points: List[Point],
    anchor_200: float,
    anchor_final: float,
    scale: float,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Candidate Group IoU vs Search Cost, Calibrated by 200 Epochs",
        "",
        f"- Anchor: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10`.",
        f"- Anchor best IoU within 200 epochs: `{anchor_200:.4f}`.",
        f"- Anchor final/best IoU: `{anchor_final:.4f}`.",
        f"- Mapping factor: `{anchor_final:.6f} / {anchor_200:.6f} = {scale:.6f}`.",
        "- Mapped IoU formula: `candidate_200epoch_best_iou * mapping_factor`.",
        "- Search cost is parsed from `logs/train_console.txt` by summing logged average batch time for detected epochs and multiplying by GPU count.",
        "- The plotted right axis uses warmup + search GPU-hours; search-only GPU-hours are also saved in CSV.",
        "",
        "## Plotted Points",
        "",
        "| Candidate group | 200ep IoU | mapped IoU | logged epochs | search cost GPU-h | cost source |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for p in points:
        lines.append(
            f"| `{p.group}` | {p.effect_200_iou:.4f} | {p.mapped_iou:.4f} | "
            f"{p.total_logged_epochs} | {p.total_gpu_hours:.4f} | `{p.search_cost_source.parent.parent.name}` |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            f"/home/intern/anaconda3/envs/new_env/bin/python {Path(__file__).resolve()} "
            f"--logs-root {args.logs_root.resolve()} --output-dir {args.output_dir.resolve()}",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("search/logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("search/figures"))
    parser.add_argument(
        "--search-only-cost",
        action="store_true",
        help="Plot only non-warmup search epochs on the right axis. CSV always contains both.",
    )
    args = parser.parse_args()

    points, anchor_200, anchor_final, scale = build_points(args.logs_root, not args.search_only_cost)

    suffix = "calibrated_200epoch"
    if args.search_only_cost:
        suffix += "_search_only"
    csv_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.csv"
    png_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.png"
    pdf_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.pdf"
    md_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.md"

    write_csv(csv_path, points)
    plot(points, png_path, pdf_path, not args.search_only_cost)
    write_report(md_path, points, anchor_200, anchor_final, scale, args)

    print(f"anchor_200_iou: {anchor_200:.6f}")
    print(f"anchor_final_iou: {anchor_final:.6f}")
    print(f"mapping_factor: {scale:.6f}")
    print(f"saved csv: {csv_path}")
    print(f"saved png: {png_path}")
    print(f"saved pdf: {pdf_path}")
    print(f"saved report: {md_path}")


if __name__ == "__main__":
    main()
