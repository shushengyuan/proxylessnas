#!/usr/bin/env python3
"""Plot candidate-group IoU and search cost from ProxylessNAS search logs."""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CANDIDATE_COUNTS: Dict[str, int] = {
    "ResConv": 3,
    "Res_Group": 6,
    "Res_Group_Spa": 9,
    "Res_Group_Spa_MBConv": 12,
    "whole": 12,
    "whole_all": 18,
    "Shuffle": 3,
    "Ghost": 3,
    "Shuffle_Ghost": 6,
}

GROUP_ORDER = [
    "ResConv",
    "Res_Group",
    "Res_Group_Spa",
    "Res_Group_Spa_MBConv",
    "whole",
    "whole_all",
]

GROUP_LABELS = {
    "ResConv": "Res",
    "Res_Group": "Res+Group",
    "Res_Group_Spa": "Res+Group+Spa",
    "Res_Group_Spa_MBConv": "Res+Group+Spa+MBConv",
    "whole": "Whole",
    "whole_all": "Whole+Shuffle+Ghost",
}

IOU_PATTERNS = [
    re.compile(r"val_mean_IOU\s+([0-9.]+)\s*\(([0-9.]+)\)", re.IGNORECASE),
    re.compile(r"Validate_IoU\s+([0-9.]+)\s*\(([0-9.]+)\)", re.IGNORECASE),
    re.compile(r"validate_IoU\s+([0-9.]+)\s*\(([0-9.]+)\)", re.IGNORECASE),
]

TRAIN_TIME_PATTERN = re.compile(
    r"^(Warmup\s+)?Train\s+\[(\d+)\]\[(\d+)/(\d+)\]\s+"
    r"Time\s+[0-9.]+\s+\(([0-9.]+)\)"
)


@dataclass
class SearchRun:
    run_name: str
    run_dir: Path
    candidate_group: str
    candidate_count: int
    target_hardware: str
    gpu: str
    gpu_count: int
    warmup_epochs: Optional[int]
    search_epochs: Optional[int]
    search_iou: Optional[float]
    iou_records: int
    train_wall_hours: Optional[float]
    train_epoch_records: int
    search_gpu_hours: Optional[float]
    has_valid_log: bool
    has_train_log: bool
    is_ablation: bool

    def to_csv_row(self) -> Dict[str, str]:
        return {
            "candidate_group": self.candidate_group,
            "candidate_count": str(self.candidate_count),
            "target_hardware": self.target_hardware,
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "gpu": self.gpu,
            "gpu_count": str(self.gpu_count),
            "warmup_epochs": _fmt_optional_int(self.warmup_epochs),
            "search_epochs": _fmt_optional_int(self.search_epochs),
            "search_iou": _fmt_optional_float(self.search_iou, 6),
            "iou_records": str(self.iou_records),
            "train_wall_hours": _fmt_optional_float(self.train_wall_hours, 6),
            "train_epoch_records": str(self.train_epoch_records),
            "search_gpu_hours": _fmt_optional_float(self.search_gpu_hours, 6),
            "has_valid_log": str(self.has_valid_log),
            "has_train_log": str(self.has_train_log),
            "is_ablation": str(self.is_ablation),
        }


def _fmt_optional_int(value: Optional[int]) -> str:
    return "" if value is None else str(value)


def _fmt_optional_float(value: Optional[float], precision: int = 4) -> str:
    return "" if value is None else f"{value:.{precision}f}"


def _parse_int(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def read_parameters(path: Path) -> Dict[str, str]:
    params: Dict[str, str] = {}
    if not path.is_file():
        return params
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":--" not in line:
            continue
        key, value = line.split(":--", 1)
        params[key.strip()] = value.strip()
    return params


def count_gpus(gpu_value: str) -> int:
    gpus = [item.strip() for item in str(gpu_value or "").split(",") if item.strip()]
    return max(1, len(gpus))


def parse_best_iou(path: Path) -> Tuple[Optional[float], int]:
    if not path.is_file():
        return None, 0
    best_iou: Optional[float] = None
    n_records = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for pattern in IOU_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            current_or_best = float(match.group(2))
            best_iou = current_or_best if best_iou is None else max(best_iou, current_or_best)
            n_records += 1
            break
    return best_iou, n_records


def parse_train_wall_hours(path: Path) -> Tuple[Optional[float], int]:
    """Estimate train-loop wall hours from logged per-batch epoch averages."""

    if not path.is_file():
        return None, 0

    final_epoch_rows: Dict[Tuple[str, int], Tuple[int, int, float]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_TIME_PATTERN.search(line)
        if not match:
            continue
        phase = "warmup" if match.group(1) else "search"
        epoch = int(match.group(2))
        batch_idx = int(match.group(3))
        last_batch_idx = int(match.group(4))
        avg_batch_seconds = float(match.group(5))
        key = (phase, epoch)
        old_row = final_epoch_rows.get(key)
        if old_row is None or batch_idx >= old_row[0]:
            final_epoch_rows[key] = (batch_idx, last_batch_idx, avg_batch_seconds)

    if not final_epoch_rows:
        return None, 0

    total_seconds = 0.0
    for _, last_batch_idx, avg_batch_seconds in final_epoch_rows.values():
        total_seconds += avg_batch_seconds * (last_batch_idx + 1)
    return total_seconds / 3600.0, len(final_epoch_rows)


def collect_runs(logs_root: Path, target_hardware: str, include_ablation: bool) -> List[SearchRun]:
    runs: List[SearchRun] = []
    for run_dir in sorted(logs_root.iterdir()):
        if not run_dir.is_dir() or "_Retrain" in run_dir.name:
            continue
        if run_dir.name.startswith("ablate_") and not include_ablation:
            continue

        params = read_parameters(run_dir / "parameters.txt")
        candidate_group = params.get("candidates_type")
        if not candidate_group or candidate_group not in CANDIDATE_COUNTS:
            continue

        hardware = params.get("target_hardware", "")
        if target_hardware != "all" and hardware != target_hardware:
            continue

        valid_log = run_dir / "logs" / "valid_console.txt"
        train_log = run_dir / "logs" / "train_console.txt"
        search_iou, iou_records = parse_best_iou(valid_log)
        train_wall_hours, train_epoch_records = parse_train_wall_hours(train_log)
        gpu_value = params.get("gpu", "")
        gpu_count = count_gpus(gpu_value)
        search_gpu_hours = None
        if train_wall_hours is not None:
            search_gpu_hours = train_wall_hours * gpu_count

        runs.append(
            SearchRun(
                run_name=run_dir.name,
                run_dir=run_dir.resolve(),
                candidate_group=candidate_group,
                candidate_count=CANDIDATE_COUNTS[candidate_group],
                target_hardware=hardware,
                gpu=gpu_value,
                gpu_count=gpu_count,
                warmup_epochs=_parse_int(params.get("warmup_epochs")),
                search_epochs=_parse_int(params.get("n_epochs")),
                search_iou=search_iou,
                iou_records=iou_records,
                train_wall_hours=train_wall_hours,
                train_epoch_records=train_epoch_records,
                search_gpu_hours=search_gpu_hours,
                has_valid_log=valid_log.is_file(),
                has_train_log=train_log.is_file(),
                is_ablation=run_dir.name.startswith("ablate_"),
            )
        )
    return runs


def is_usable_for_plot(run: SearchRun) -> bool:
    if run.search_iou is None or run.search_gpu_hours is None:
        return False
    if not run.search_epochs or run.search_epochs <= 0:
        return False
    if run.iou_records <= 0 or run.train_epoch_records <= 0:
        return False
    return True


def group_sort_key(group: str) -> Tuple[int, int, str]:
    if group in GROUP_ORDER:
        return (0, GROUP_ORDER.index(group), group)
    return (1, CANDIDATE_COUNTS.get(group, 10_000), group)


def select_best_by_group(runs: Iterable[SearchRun]) -> List[SearchRun]:
    best: Dict[str, SearchRun] = {}
    for run in runs:
        if not is_usable_for_plot(run):
            continue
        old = best.get(run.candidate_group)
        if old is None:
            best[run.candidate_group] = run
            continue
        # Primary: best IoU. Secondary: lower GPU-hours if IoU ties.
        assert run.search_iou is not None and old.search_iou is not None
        assert run.search_gpu_hours is not None and old.search_gpu_hours is not None
        if (run.search_iou, -run.search_gpu_hours) > (old.search_iou, -old.search_gpu_hours):
            best[run.candidate_group] = run
    return sorted(best.values(), key=lambda row: group_sort_key(row.candidate_group))


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(best_rows: List[SearchRun], output_png: Path, output_pdf: Path, title: str) -> None:
    if not best_rows:
        raise RuntimeError("no usable rows to plot")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(best_rows)))
    labels = [
        f"{GROUP_LABELS.get(row.candidate_group, row.candidate_group)}\n({row.candidate_count} ops)"
        for row in best_rows
    ]
    ious = [float(row.search_iou) for row in best_rows]
    costs = [float(row.search_gpu_hours) for row in best_rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )
    fig, ax_iou = plt.subplots(figsize=(9.2, 5.2), dpi=220)
    ax_cost = ax_iou.twinx()

    iou_color = "#1f5f99"
    cost_color = "#c96f1a"
    ax_iou.plot(xs, ious, marker="o", markersize=8, linewidth=2.4, color=iou_color, label="Best search IoU")
    ax_cost.plot(
        xs,
        costs,
        marker="s",
        markersize=7,
        linewidth=2.2,
        linestyle="--",
        color=cost_color,
        label="Search cost",
    )

    ax_iou.set_xticks(xs)
    ax_iou.set_xticklabels(labels)
    ax_iou.set_xlabel("Search Candidate Group")
    ax_iou.set_ylabel("IoU", color=iou_color)
    ax_cost.set_ylabel("Search Cost (GPU-hours)", color=cost_color)
    ax_iou.tick_params(axis="y", labelcolor=iou_color)
    ax_cost.tick_params(axis="y", labelcolor=cost_color)

    y_min = max(0.0, min(ious) - 0.035)
    y_max = min(1.0, max(ious) + 0.035)
    if math.isclose(y_min, y_max):
        y_min = max(0.0, y_min - 0.05)
        y_max = min(1.0, y_max + 0.05)
    ax_iou.set_ylim(y_min, y_max)
    ax_cost.set_ylim(0, max(costs) * 1.25 if costs else 1)

    ax_iou.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.65)
    ax_iou.set_axisbelow(True)
    ax_iou.set_title(title)

    for x, y, row in zip(xs, ious, best_rows):
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


def write_report(path: Path, raw_rows: List[SearchRun], best_rows: List[SearchRun], args: argparse.Namespace) -> None:
    missing_groups = []
    seen_groups = {row.candidate_group for row in raw_rows}
    plotted_groups = {row.candidate_group for row in best_rows}
    for group in sorted(seen_groups, key=group_sort_key):
        if group not in plotted_groups:
            missing_groups.append(group)

    lines = [
        "# Candidate Group IoU vs Search Cost",
        "",
        f"- Logs root: `{args.logs_root.resolve()}`",
        f"- Target hardware filter: `{args.target_hardware}`",
        f"- Include ablation dirs: `{args.include_ablation}`",
        "- IoU: best value in `logs/valid_console.txt` from `val_mean_IOU` / `Validate_IoU`.",
        "- Search cost: sum of final logged epoch average batch time times batch count, then multiplied by GPU count to get GPU-hours.",
        "- Note: the cost is log-estimated train-loop GPU-hours; validation and checkpoint overhead are not included.",
        "",
        "## Plotted Points",
        "",
        "| Candidate group | Ops | IoU | Search cost (GPU-h) | Wall hours | Source run |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in best_rows:
        lines.append(
            f"| `{row.candidate_group}` | {row.candidate_count} | "
            f"{row.search_iou:.4f} | {row.search_gpu_hours:.4f} | "
            f"{row.train_wall_hours:.4f} | `{row.run_name}` |"
        )

    if missing_groups:
        lines.extend(["", "## Not Plotted", ""])
        for group in missing_groups:
            group_rows = [row for row in raw_rows if row.candidate_group == group]
            reasons = []
            if all(row.search_iou is None for row in group_rows):
                reasons.append("no valid IoU log")
            if all(row.search_gpu_hours is None for row in group_rows):
                reasons.append("no train-time log")
            if all(not row.search_epochs or row.search_epochs <= 0 for row in group_rows):
                reasons.append("search_epochs <= 0")
            reason_text = ", ".join(reasons) if reasons else "not selected as group best"
            lines.append(f"- `{group}`: {reason_text}.")

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            f"/home/intern/anaconda3/envs/new_env/bin/python {Path(__file__).resolve()} "
            f"--logs-root {args.logs_root.resolve()} --output-dir {args.output_dir.resolve()} "
            f"--target-hardware {args.target_hardware}",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, default=Path("search/logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("search/figures"))
    parser.add_argument("--target-hardware", choices=["gpu", "cpu", "flops", "all"], default="gpu")
    parser.add_argument("--include-ablation", action="store_true")
    args = parser.parse_args()

    runs = collect_runs(args.logs_root, args.target_hardware, args.include_ablation)
    best_rows = select_best_by_group(runs)

    suffix = args.target_hardware
    if args.include_ablation:
        suffix += "_with_ablation"

    raw_csv = args.output_dir / f"candidate_group_iou_search_cost_raw_{suffix}.csv"
    best_csv = args.output_dir / f"candidate_group_iou_search_cost_best_{suffix}.csv"
    output_png = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.png"
    output_pdf = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.pdf"
    report_md = args.output_dir / f"candidate_group_iou_search_cost_{suffix}.md"

    write_csv(raw_csv, [row.to_csv_row() for row in runs])
    write_csv(best_csv, [row.to_csv_row() for row in best_rows])
    plot(
        best_rows,
        output_png,
        output_pdf,
        f"Candidate Group vs IoU and Search Cost ({args.target_hardware})",
    )
    write_report(report_md, runs, best_rows, args)

    print(f"saved raw csv: {raw_csv}")
    print(f"saved best csv: {best_csv}")
    print(f"saved png: {output_png}")
    print(f"saved pdf: {output_pdf}")
    print(f"saved report: {report_md}")


if __name__ == "__main__":
    main()
