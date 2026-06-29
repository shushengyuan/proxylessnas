#!/usr/bin/env python3
"""Plot candidate-group curve from 400-epoch search samples."""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


VALID_PATTERN = re.compile(
    r"^(?:==========)?(Warmup\s+)?Valid\s+\[(\d+)/(\d+)\].*?"
    r"(?:val_mean_IOU|Validate_IoU|validate_IoU)\s+([0-9.]+)\s*\(([0-9.]+)\)",
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
    cost_log: str
    gpu_count: int
    source_type: str


@dataclass
class Point:
    group: str
    label: str
    ops: int
    effect_400_iou: float
    effect_best_epoch: int
    mapped_iou: float
    cost_wall_hours_400: float
    cost_gpu_hours_400: float
    cost_epochs: int
    gpu_count: int
    effect_source: Path
    cost_source: Path
    source_type: str


POINT_SPECS = [
    PointSpec(
        "GroupConv",
        "Group",
        3,
        "curve_groupconv_400.console.log",
        "curve_groupconv_400.console.log",
        1,
        "new 400ep search",
    ),
    PointSpec(
        "Group_Spa",
        "Group+Spa",
        6,
        "curve_groupspa_400.console.log",
        "curve_groupspa_400.console.log",
        1,
        "new 400ep search",
    ),
    PointSpec(
        "Group_Spa_Res",
        "Group+Spa+Res",
        9,
        "curve_groupspares_400.console.log",
        "curve_groupspares_400.console.log",
        1,
        "new 400ep search",
    ),
    PointSpec(
        "Group_Spa_Res_MBConv",
        "Group+Spa+Res+MB",
        12,
        "0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/logs/valid_console.txt",
        "0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00/logs/train_console.txt",
        2,
        "existing Res_Group_Spa_MBConv search, first 400 search epochs",
    ),
    PointSpec(
        "whole",
        "Whole+Shuffle+Ghost",
        18,
        "ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/logs/valid_console.txt",
        "0,1_NUAA-SIRST_Super_all_whole_all_27_04_2026_00_04_38/logs/train_console.txt",
        2,
        "best full retrain IoU anchor + existing full search cost",
    ),
]


def parse_best_iou(valid_log: Path, max_epoch: int) -> Tuple[float, int, int]:
    best_iou = None
    best_epoch = None
    n_records = 0
    for line in valid_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = VALID_PATTERN.search(line)
        if not match:
            continue
        phase = "warmup" if match.group(1) else "train"
        if phase != "train":
            continue
        epoch = int(match.group(2))
        if epoch > max_epoch:
            continue
        best_value = float(match.group(5))
        n_records += 1
        if best_iou is None or best_value > best_iou:
            best_iou = best_value
            best_epoch = epoch
    if best_iou is None or best_epoch is None:
        raise RuntimeError(f"no valid IoU records in first {max_epoch} train epochs: {valid_log}")
    return best_iou, best_epoch, n_records


def parse_search_cost(train_log: Path, max_epoch: int) -> Tuple[float, int]:
    rows: Dict[int, Tuple[int, int, float]] = {}
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_TIME_PATTERN.search(line)
        if not match:
            continue
        phase = "warmup" if match.group(1) else "search"
        if phase != "search":
            continue
        epoch = int(match.group(2))
        if epoch > max_epoch:
            continue
        batch_idx = int(match.group(3))
        last_batch_idx = int(match.group(4))
        avg_batch_seconds = float(match.group(5))
        old = rows.get(epoch)
        if old is None or batch_idx >= old[0]:
            rows[epoch] = (batch_idx, last_batch_idx, avg_batch_seconds)
    if not rows:
        raise RuntimeError(f"no train time rows in first {max_epoch} search epochs: {train_log}")
    total_seconds = sum(avg * (last_idx + 1) for _, last_idx, avg in rows.values())
    return total_seconds / 3600.0, len(rows)


def build_points(logs_root: Path, max_epoch: int) -> Tuple[List[Point], float, float, float]:
    anchor_log = logs_root / "ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10/logs/valid_console.txt"
    anchor_400, _, _ = parse_best_iou(anchor_log, max_epoch)
    anchor_final, _, _ = parse_best_iou(anchor_log, 10_000)
    mapping_factor = anchor_final / anchor_400

    points = []
    for spec in POINT_SPECS:
        effect_source = logs_root / spec.effect_log
        cost_source = logs_root / spec.cost_log
        effect_iou, best_epoch, n_records = parse_best_iou(effect_source, max_epoch)
        if n_records < max_epoch:
            raise RuntimeError(f"{spec.group} has only {n_records}/{max_epoch} IoU records: {effect_source}")
        wall_hours, cost_epochs = parse_search_cost(cost_source, max_epoch)
        if cost_epochs < max_epoch:
            raise RuntimeError(f"{spec.group} has only {cost_epochs}/{max_epoch} cost epochs: {cost_source}")
        points.append(
            Point(
                group=spec.group,
                label=spec.label,
                ops=spec.ops,
                effect_400_iou=effect_iou,
                effect_best_epoch=best_epoch,
                mapped_iou=effect_iou * mapping_factor,
                cost_wall_hours_400=wall_hours,
                cost_gpu_hours_400=wall_hours * spec.gpu_count,
                cost_epochs=cost_epochs,
                gpu_count=spec.gpu_count,
                effect_source=effect_source.resolve(),
                cost_source=cost_source.resolve(),
                source_type=spec.source_type,
            )
        )
    return points, anchor_400, anchor_final, mapping_factor


def write_csv(path: Path, points: List[Point]) -> None:
    rows = []
    for p in points:
        rows.append(
            {
                "candidate_group": p.group,
                "label": p.label,
                "ops": p.ops,
                "effect_400_iou": f"{p.effect_400_iou:.6f}",
                "effect_best_epoch": p.effect_best_epoch,
                "mapped_iou": f"{p.mapped_iou:.6f}",
                "search_cost_wall_hours_400": f"{p.cost_wall_hours_400:.6f}",
                "gpu_count": p.gpu_count,
                "search_cost_gpu_hours_400": f"{p.cost_gpu_hours_400:.6f}",
                "cost_epochs": p.cost_epochs,
                "effect_source": str(p.effect_source),
                "cost_source": str(p.cost_source),
                "source_type": p.source_type,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(points: List[Point], output_png: Path, output_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(points)))
    labels = [f"{p.label}\n({p.ops} ops)" for p in points]
    ious = [p.mapped_iou for p in points]
    costs = [p.cost_gpu_hours_400 for p in points]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax_iou = plt.subplots(figsize=(11.5, 5.6), dpi=220)
    ax_cost = ax_iou.twinx()

    iou_color = "#165a8f"
    cost_color = "#c86913"
    ax_iou.plot(xs, ious, marker="o", markersize=7, linewidth=2.5, color=iou_color, label="Mapped IoU")
    ax_cost.plot(
        xs,
        costs,
        marker="s",
        markersize=6.5,
        linewidth=2.3,
        linestyle="--",
        color=cost_color,
        label="Search cost",
    )

    ax_iou.set_xticks(xs)
    ax_iou.set_xticklabels(labels)
    ax_iou.set_xlabel("Search Candidate Group")
    ax_iou.set_ylabel("Mapped IoU", color=iou_color)
    ax_cost.set_ylabel("Search Cost (GPU-hours, first 400 search epochs)", color=cost_color)
    ax_iou.tick_params(axis="y", labelcolor=iou_color)
    ax_cost.tick_params(axis="y", labelcolor=cost_color)
    ax_iou.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.65)
    ax_iou.set_axisbelow(True)
    ax_iou.set_title("Candidate Group vs Mapped IoU and Search Cost (400 epochs)")

    y_min = max(0.0, min(ious) - 0.035)
    y_max = min(1.0, max(ious) + 0.035)
    if math.isclose(y_min, y_max):
        y_min = max(0.0, y_min - 0.05)
        y_max = min(1.0, y_max + 0.05)
    ax_iou.set_ylim(y_min, y_max)
    ax_cost.set_ylim(0, max(costs) * 1.25)

    for x, y, p in zip(xs, ious, points):
        ax_iou.annotate(
            f"{y:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            color=iou_color,
            fontsize=9,
            fontweight="bold",
        )
        ax_iou.annotate(
            f"400ep {p.effect_400_iou:.3f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 23),
            ha="center",
            color=iou_color,
            fontsize=8,
        )
    for x, y in zip(xs, costs):
        ax_cost.annotate(
            f"{y:.2f}",
            (x, y),
            textcoords="offset points",
            xytext=(0, -16),
            ha="center",
            color=cost_color,
            fontsize=9,
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
    anchor_400: float,
    anchor_final: float,
    mapping_factor: float,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Candidate Group IoU vs Search Cost, 400-Epoch Mapping",
        "",
        f"- Anchor folder: `/home/intern/proxylessnas/search/logs/ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10`.",
        f"- Anchor best IoU within 400 epochs: `{anchor_400:.4f}`.",
        f"- Anchor final/best IoU: `{anchor_final:.4f}`.",
        f"- Mapping factor: `{anchor_final:.6f} / {anchor_400:.6f} = {mapping_factor:.6f}`.",
        "- Mapped IoU formula: `candidate_400epoch_best_iou * mapping_factor`.",
        "- Search cost: first 400 non-warmup search epochs, parsed from logged average batch time and multiplied by GPU count.",
        "",
        "## Points",
        "",
        "| Candidate group | Ops | 400ep IoU | mapped IoU | best epoch | search cost GPU-h | source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for p in points:
        lines.append(
            f"| `{p.group}` | {p.ops} | {p.effect_400_iou:.4f} | {p.mapped_iou:.4f} | "
            f"{p.effect_best_epoch} | {p.cost_gpu_hours_400:.4f} | {p.source_type} |"
        )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "Run missing redesigned 3/6/9-op experiments if logs are absent:",
            "",
            "```bash",
            "bash search/run_candidate_group_400epoch_search.sh GroupConv 4 400epoch_curve_groupconv | tee search/logs/curve_groupconv_400.console.log",
            "bash search/run_candidate_group_400epoch_search.sh Group_Spa 7 400epoch_curve_groupspa | tee search/logs/curve_groupspa_400.console.log",
            "bash search/run_candidate_group_400epoch_search.sh Group_Spa_Res 6 400epoch_curve_groupspares | tee search/logs/curve_groupspares_400.console.log",
            "```",
            "",
            "Generate the plot:",
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
    parser.add_argument("--max-epoch", type=int, default=400)
    args = parser.parse_args()

    points, anchor_400, anchor_final, mapping_factor = build_points(args.logs_root, args.max_epoch)
    suffix = f"400epoch_mapped"
    csv_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.csv"
    png_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.png"
    pdf_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.pdf"
    md_path = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.md"

    write_csv(csv_path, points)
    plot(points, png_path, pdf_path)
    write_report(md_path, points, anchor_400, anchor_final, mapping_factor, args)

    print(f"anchor_400_iou: {anchor_400:.6f}")
    print(f"anchor_final_iou: {anchor_final:.6f}")
    print(f"mapping_factor: {mapping_factor:.6f}")
    print(f"saved csv: {csv_path}")
    print(f"saved png: {png_path}")
    print(f"saved pdf: {pdf_path}")
    print(f"saved report: {md_path}")


if __name__ == "__main__":
    main()
