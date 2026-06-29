#!/usr/bin/env python3
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional


def _last_match(lines, pattern: re.Pattern[str]) -> Optional[str]:
    value = None
    for line in lines:
        m = pattern.search(line)
        if m:
            value = m.group(1).strip()
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract final genotype from pt.darts search log.")
    parser.add_argument("--run-dir", type=Path, required=True, help="pt.darts search run dir (searchs/<name>).")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise SystemExit(f"run dir not found: {run_dir}")

    log_files = sorted(run_dir.glob("*.log"))
    if not log_files:
        raise SystemExit(f"no .log files found under: {run_dir}")
    log_path = log_files[-1]

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    best_genotype = _last_match(lines, re.compile(r"Best Genotype = (.+)$"))
    last_genotype = _last_match(lines, re.compile(r"genotype = (.+)$"))
    best_top1 = _last_match(lines, re.compile(r"Final best Prec@1 = ([0-9.]+%)"))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "log_file": str(log_path),
        "best_top1": best_top1,
        "best_genotype": best_genotype,
        "last_epoch_genotype": last_genotype,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
