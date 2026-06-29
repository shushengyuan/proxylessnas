#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_proxyless_run(payload: Dict[str, Any], run_name: Optional[str]) -> Dict[str, Any]:
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("proxyless payload has no runs")
    if run_name:
        for row in runs:
            if row.get("run_name") == run_name:
                return row
        raise ValueError(f"proxyless run_name not found: {run_name}")
    completed = [r for r in runs if r.get("status") == "completed"]
    if completed:
        return sorted(completed, key=lambda x: x.get("run_name", ""))[-1]
    return runs[-1]


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    return str(v)


def _write_markdown(
    out_path: Path,
    proxyless_run: Dict[str, Any],
    darts_payload: Optional[Dict[str, Any]],
) -> None:
    p_budget = proxyless_run.get("search_budget", {})
    p_reg = proxyless_run.get("regularization", {})
    p_arch = proxyless_run.get("architecture", {})
    p_metrics = proxyless_run.get("metrics", {})

    darts_genotype = "N/A"
    darts_top1 = "N/A"
    darts_run_dir = "N/A"
    if darts_payload:
        darts_genotype = _fmt(darts_payload.get("best_genotype") or darts_payload.get("last_epoch_genotype"))
        darts_top1 = _fmt(darts_payload.get("best_top1"))
        darts_run_dir = _fmt(darts_payload.get("run_dir"))

    lines: List[str] = []
    lines.append("# DARTS vs ProxylessNAS: Search-Only Comparison")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Proxyless run: `{proxyless_run.get('run_name')}`")
    lines.append("")
    lines.append("## Search Setup Snapshot")
    lines.append("")
    lines.append("| Dimension | ProxylessNAS | DARTS |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| Search space | Edge/block operator choices (`{_fmt(p_arch.get('candidates_type'))}`) | Cell DAG primitives |")
    lines.append(f"| Proxy task | `{_fmt(proxyless_run.get('dataset'))}` | `cifar10` |")
    lines.append(f"| Optimization | Gradient + binary path (`{_fmt(p_reg.get('reg_type'))}`) | Continuous alpha + bilevel (unrolled) |")
    lines.append(f"| Epoch budget | warmup `{_fmt(p_budget.get('warmup_epochs'))}` + search `{_fmt(p_budget.get('search_epochs'))}` | see run dir `{darts_run_dir}` |")
    lines.append(f"| Batch size | train `{_fmt(proxyless_run.get('batch_size', {}).get('train'))}` | from DARTS run log |")
    lines.append("")
    lines.append("## Discovered Architecture Artifacts")
    lines.append("")
    lines.append(f"- Proxyless net config: `{_fmt(proxyless_run.get('artifact_paths', {}).get('learned_net_config'))}`")
    lines.append(f"- Proxyless run config: `{_fmt(proxyless_run.get('artifact_paths', {}).get('learned_run_config'))}`")
    lines.append(f"- Proxyless approximate GPU-hour: `{_fmt(p_metrics.get('search_duration_hours_approx'))}`")
    lines.append(f"- Proxyless GPU latency in search log: `{_fmt(p_metrics.get('gpu_avg_time_ms_from_log'))} ms`")
    lines.append(f"- DARTS best top1 (proxy metric): `{darts_top1}`")
    lines.append(f"- DARTS best genotype: `{darts_genotype}`")
    lines.append("")
    lines.append("## Conclusion Boundary")
    lines.append("")
    lines.append("- This report compares **search behavior and searched structures**, not downstream task metrics.")
    lines.append("- If you need strict task-level fairness, run a DARTS-style search on the same SIRST pipeline.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown report for DARTS vs Proxyless search-only comparison.")
    parser.add_argument("--proxyless-json", type=Path, required=True)
    parser.add_argument("--proxyless-run-name", type=str, default=None)
    parser.add_argument("--darts-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    proxyless_payload = _load_json(args.proxyless_json)
    proxyless_run = _pick_proxyless_run(proxyless_payload, args.proxyless_run_name)
    darts_payload = _load_json(args.darts_json) if args.darts_json and args.darts_json.is_file() else None
    _write_markdown(args.output_md, proxyless_run, darts_payload)
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
