#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def _coerce_scalar(text: str) -> Any:
    v = text.strip()
    if v in {"None", ""}:
        return None
    if v in {"True", "False"}:
        return v == "True"
    try:
        if "." in v or "e" in v or "E" in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        out: Dict[str, Any] = {}
        for line in raw.splitlines():
            if ":--" not in line:
                continue
            key, value = line.split(":--", 1)
            out[key.strip()] = _coerce_scalar(value)
        return out


def _parse_gpu_ms_from_log(train_log: Path) -> Optional[float]:
    if not train_log.is_file():
        return None
    pattern = re.compile(r"gpu:\s*([0-9]+(?:\.[0-9]+)?)ms")
    last = None
    for line in train_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            last = float(match.group(1))
    return last


def _infer_search_status(run_dir: Path) -> str:
    learned = run_dir / "learned_net"
    if (learned / "net.config").is_file() and (learned / "run.config").is_file():
        return "completed"
    if (run_dir / "checkpoint").is_dir():
        return "checkpoint_only"
    return "unknown"


def collect_one(run_dir: Path) -> Dict[str, Any]:
    params = _read_json(run_dir / "parameters.txt")
    run_cfg = _read_json(run_dir / "learned_net" / "run.config")
    net_cfg = _read_json(run_dir / "learned_net" / "net.config")
    train_log = run_dir / "logs" / "train_console.txt"

    start_ts = datetime.fromtimestamp(run_dir.stat().st_mtime)
    end_ts = start_ts
    if train_log.is_file():
        end_ts = datetime.fromtimestamp(train_log.stat().st_mtime)

    search_hours = max(0.0, (end_ts - start_ts).total_seconds() / 3600.0)
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "status": _infer_search_status(run_dir),
        "seed": params.get("manual_seed"),
        "dataset": params.get("dataset", run_cfg.get("dataset")),
        "gpu": params.get("gpu"),
        "batch_size": {
            "train": params.get("train_batch_size", run_cfg.get("train_batch_size")),
            "test": params.get("test_batch_size", run_cfg.get("test_batch_size")),
        },
        "search_budget": {
            "warmup_epochs": params.get("warmup_epochs"),
            "search_epochs": params.get("n_epochs", run_cfg.get("n_epochs")),
            "grad_update_every": params.get("grad_update_arch_param_every"),
            "grad_update_steps": params.get("grad_update_steps"),
        },
        "regularization": {
            "target_hardware": params.get("target_hardware"),
            "reg_type": params.get("grad_reg_loss_type"),
            "reg_lambda": params.get("grad_reg_loss_lambda"),
            "reg_alpha": params.get("grad_reg_loss_alpha"),
            "reg_beta": params.get("grad_reg_loss_beta"),
            "ref_value": params.get("ref_value"),
        },
        "architecture": {
            "model_name": params.get("model_name"),
            "candidates_type": params.get("candidates_type"),
            "iterations": params.get("iterations"),
            "channel_num": params.get("channel_num"),
            "conv_type": params.get("conv_type"),
            "net_name": net_cfg.get("name"),
            "net_blocks": len(net_cfg.get("blocks", [])) if isinstance(net_cfg.get("blocks"), list) else None,
        },
        "artifact_paths": {
            "parameters_txt": str(run_dir / "parameters.txt"),
            "learned_net_config": str(run_dir / "learned_net" / "net.config"),
            "learned_run_config": str(run_dir / "learned_net" / "run.config"),
            "arch_log": str(run_dir / "logs" / "arch.log"),
            "train_log": str(train_log),
        },
        "metrics": {
            "gpu_avg_time_ms_from_log": _parse_gpu_ms_from_log(train_log),
            "search_duration_hours_approx": round(search_hours, 4),
        },
    }


def _is_search_run(name: str) -> bool:
    return "_Retrain" not in name and "_Super_" in name


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ProxylessNAS search artifacts.")
    parser.add_argument("--logs-root", type=Path, required=True, help="Root folder containing search run dirs.")
    parser.add_argument("--run-name", type=str, default=None, help="Specific run directory name to extract.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    root = args.logs_root
    if not root.is_dir():
        raise SystemExit(f"logs root does not exist: {root}")

    if args.run_name:
        target = root / args.run_name
        if not target.is_dir():
            raise SystemExit(f"run not found: {target}")
        runs = [target]
    else:
        runs = sorted([p for p in root.iterdir() if p.is_dir() and _is_search_run(p.name)])
        if not runs:
            raise SystemExit(f"no search runs found under: {root}")

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "logs_root": str(root),
        "count": len(runs),
        "runs": [collect_one(p) for p in runs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
