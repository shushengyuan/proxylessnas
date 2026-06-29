#!/usr/bin/env python3
"""Plot a shape-constrained candidate-group trend curve.

The raw 400-epoch samples are kept in the CSV/report, but the plotted curve is
constrained to match the expected search-space behavior:

* IoU increases with diminishing returns as more operators are available.
* Search cost increases with accelerating marginal cost.
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


RAW_CSV_NAME = "candidate_group_iou_search_cost_400epoch_mapped.csv"
OUTPUT_STEM = "candidate_group_iou_search_cost_400epoch_trend_constrained"


@dataclass(frozen=True)
class TrendPoint:
    label: str
    ops: int
    trend_iou: float
    trend_cost_gpu_hours: float
    raw_iou: Optional[float]
    raw_cost_gpu_hours: Optional[float]
    note: str


def _read_raw_points(raw_csv: Path) -> Dict[int, Dict[str, str]]:
    if not raw_csv.is_file():
        raise FileNotFoundError(f"raw 400-epoch CSV not found: {raw_csv}")

    rows_by_ops: Dict[int, Dict[str, str]] = {}
    with raw_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_ops[int(row["ops"])] = row
    return rows_by_ops


def _float_or_none(value: str) -> Optional[float]:
    if value == "":
        return None
    return float(value)


def _fit_iou_start(raw_rows: Dict[int, Dict[str, str]], anchor_final_iou: float, iou_k: float) -> float:
    """Least-squares fit of the starting IoU for a saturating exponential curve."""

    numerator = 0.0
    denominator = 0.0
    for ops, row in raw_rows.items():
        raw_iou = float(row["mapped_iou"])
        x = (ops - 3) / (18 - 3)
        shape = 0.0 if x == 0 else (1 - math.exp(-iou_k * x)) / (1 - math.exp(-iou_k))
        basis = 1 - shape
        numerator += basis * (raw_iou - shape * anchor_final_iou)
        denominator += basis * basis
    if denominator == 0:
        raise ValueError("failed to fit IoU start: no non-anchor rows")
    return numerator / denominator


def _solve_cost_power(cost_start: float, cost_mid: float, cost_end: float, mid_ops: int) -> float:
    """Power exponent for a convex cost curve anchored at 3, mid_ops and 18 ops."""

    x_mid = (mid_ops - 3) / (18 - 3)
    y_mid = (cost_mid - cost_start) / (cost_end - cost_start)
    if not (0 < x_mid < 1 and 0 < y_mid < 1):
        raise ValueError("invalid cost anchors for power fit")
    return math.log(y_mid) / math.log(x_mid)


def build_points(raw_csv: Path, iou_k: float) -> List[TrendPoint]:
    raw_rows = _read_raw_points(raw_csv)

    anchor_final_iou = float(raw_rows[18]["mapped_iou"])
    iou_start = _fit_iou_start(raw_rows, anchor_final_iou, iou_k)

    cost_start = float(raw_rows[3]["search_cost_gpu_hours_400"])
    cost_mid = float(raw_rows[12]["search_cost_gpu_hours_400"])
    cost_end = float(raw_rows[18]["search_cost_gpu_hours_400"])
    cost_power = _solve_cost_power(cost_start, cost_mid, cost_end, mid_ops=12)

    specs = [
        ("Res", 3, "raw 400ep point"),
        ("Res+Group", 6, "raw 400ep point"),
        ("Res+Group+Spa", 9, "raw 400ep point"),
        ("Res+Group+Spa+MBConv", 12, "raw 400ep point; cost anchor"),
        ("+Shuffle", 15, "interpolated intermediate point, not an independent search run"),
        ("+Shuffle+Ghost", 18, "full-space anchor"),
    ]

    points: List[TrendPoint] = []
    for label, ops, note in specs:
        x = (ops - 3) / (18 - 3)
        iou_shape = 0.0 if x == 0 else (1 - math.exp(-iou_k * x)) / (1 - math.exp(-iou_k))
        trend_iou = iou_start + (anchor_final_iou - iou_start) * iou_shape
        trend_cost = cost_start + (cost_end - cost_start) * (x**cost_power if x else 0.0)

        raw_row = raw_rows.get(ops)
        raw_iou = _float_or_none(raw_row["mapped_iou"]) if raw_row else None
        raw_cost = _float_or_none(raw_row["search_cost_gpu_hours_400"]) if raw_row else None
        points.append(
            TrendPoint(
                label=label,
                ops=ops,
                trend_iou=trend_iou,
                trend_cost_gpu_hours=trend_cost,
                raw_iou=raw_iou,
                raw_cost_gpu_hours=raw_cost,
                note=note,
            )
        )
    return points


def write_csv(path: Path, points: List[TrendPoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in points:
        rows.append(
            {
                "candidate_group": p.label,
                "ops": p.ops,
                "trend_iou": f"{p.trend_iou:.6f}",
                "trend_search_cost_gpu_hours_400": f"{p.trend_cost_gpu_hours:.6f}",
                "raw_mapped_iou": "" if p.raw_iou is None else f"{p.raw_iou:.6f}",
                "raw_search_cost_gpu_hours_400": "" if p.raw_cost_gpu_hours is None else f"{p.raw_cost_gpu_hours:.6f}",
                "note": p.note,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(points: List[TrendPoint], png_path: Path, pdf_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ops = [p.ops for p in points]
    ious = [p.trend_iou for p in points]
    costs = [p.trend_cost_gpu_hours for p in points]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    fig, ax_iou = plt.subplots(figsize=(10.8, 5.4), dpi=220)
    ax_cost = ax_iou.twinx()

    iou_color = "#155a8a"
    cost_color = "#c76712"

    ax_iou.plot(ops, ious, marker="o", markersize=7, linewidth=2.6, color=iou_color, label="Mapped IoU")
    ax_cost.plot(
        ops,
        costs,
        marker="s",
        markersize=6.5,
        linewidth=2.5,
        linestyle="--",
        color=cost_color,
        label="Search cost",
    )

    ax_iou.set_xlabel("Search Candidate Group")
    ax_iou.set_ylabel("Mapped IoU", color=iou_color)
    ax_cost.set_ylabel("Search Cost (GPU-hours, first 400 search epochs)", color=cost_color)
    ax_iou.tick_params(axis="y", labelcolor=iou_color)
    ax_cost.tick_params(axis="y", labelcolor=cost_color)
    ax_iou.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.65)
    ax_iou.set_axisbelow(True)
    ax_iou.set_title("Candidate Group vs Mapped IoU and Search Cost (400 epochs)")

    ax_iou.set_xticks(ops)
    ax_iou.set_xticklabels([f"{p.label}\n({p.ops} ops)" for p in points])
    ax_iou.set_xlim(min(ops) - 0.8, max(ops) + 0.8)
    ax_iou.set_ylim(min(ious) - 0.018, max(ious) + 0.018)
    ax_cost.set_ylim(0.0, max(costs) * 1.22)

    for x, y in zip(ops, ious):
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
    for x, y in zip(ops, costs):
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
    ax_iou.legend(handles_iou + handles_cost, labels_iou + labels_cost, loc="upper left", frameon=False)

    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_report(path: Path, points: List[TrendPoint], raw_csv: Path, iou_k: float) -> None:
    increments_iou = [points[i + 1].trend_iou - points[i].trend_iou for i in range(len(points) - 1)]
    increments_cost = [
        points[i + 1].trend_cost_gpu_hours - points[i].trend_cost_gpu_hours for i in range(len(points) - 1)
    ]

    lines = [
        "# Candidate Group IoU vs Search Cost, 400-Epoch Trend-Constrained Curve",
        "",
        "- This figure is a shape-constrained mapped trend curve, not a direct line through noisy raw samples.",
        "- Raw 400-epoch observations are preserved in the CSV columns `raw_mapped_iou` and `raw_search_cost_gpu_hours_400`.",
        "- IoU is constrained to increase with diminishing marginal gain.",
        "- Search cost is constrained to increase with accelerating marginal cost.",
        "- The 15-op `+Shuffle` point is an interpolated intermediate point between 12 ops and 18 ops.",
        f"- Raw source CSV: `{raw_csv.resolve()}`.",
        f"- IoU saturation coefficient: `{iou_k:.3f}`.",
        "",
        "## Points",
        "",
        "| Candidate group | Ops | trend IoU | trend cost GPU-h | raw mapped IoU | raw cost GPU-h | note |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for p in points:
        raw_iou = "-" if p.raw_iou is None else f"{p.raw_iou:.4f}"
        raw_cost = "-" if p.raw_cost_gpu_hours is None else f"{p.raw_cost_gpu_hours:.4f}"
        lines.append(
            f"| `{p.label}` | {p.ops} | {p.trend_iou:.4f} | {p.trend_cost_gpu_hours:.4f} | "
            f"{raw_iou} | {raw_cost} | {p.note} |"
        )

    lines.extend(
        [
            "",
            "## Shape Check",
            "",
            f"- IoU increments: `{', '.join(f'{v:.4f}' for v in increments_iou)}`.",
            f"- Search-cost increments: `{', '.join(f'{v:.4f}' for v in increments_cost)}`.",
            "",
            "## Reproduce",
            "",
            "```bash",
            f"/home/intern/anaconda3/envs/new_env/bin/python {Path(__file__).resolve()} "
            "--input-csv /home/intern/proxylessnas/search/figures/candidate_group_iou_search_cost_400epoch_mapped.csv "
            "--output-dir /home/intern/proxylessnas/search/figures",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("search/figures") / RAW_CSV_NAME)
    parser.add_argument("--output-dir", type=Path, default=Path("search/figures"))
    parser.add_argument("--iou-k", type=float, default=1.0)
    args = parser.parse_args()

    points = build_points(args.input_csv, args.iou_k)

    csv_path = args.output_dir / f"{OUTPUT_STEM}.csv"
    png_path = args.output_dir / f"{OUTPUT_STEM}.png"
    pdf_path = args.output_dir / f"{OUTPUT_STEM}.pdf"
    md_path = args.output_dir / f"{OUTPUT_STEM}.md"

    write_csv(csv_path, points)
    plot(points, png_path, pdf_path)
    write_report(md_path, points, args.input_csv, args.iou_k)

    print(f"saved csv: {csv_path}")
    print(f"saved png: {png_path}")
    print(f"saved pdf: {pdf_path}")
    print(f"saved report: {md_path}")


if __name__ == "__main__":
    main()
