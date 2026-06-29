#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/intern/proxylessnas")
SEARCH = ROOT / "search"
SCRIPT = SEARCH / "retrain_whole_irstd.sh"
STATE_PATH = Path("/tmp/irstd_ablate_queue_state.json")
STATUS_PATH = Path("/tmp/irstd_ablate_queue_status.txt")
HIT_PATH = Path("/tmp/irstd_ablate_hit.txt")
TARGET = 0.6816

QUEUE = [
    "ablate_0_1_Ghost3x3_r2_nose",
    "ablate_0_1_Ghost7x7_r2_nose",
    "ablate_0_1_Ghost5x5_r2_se",
    "ablate_0_3_Shuffle3x3_e2",
    "ablate_0_3_Shuffle3x3_e8",
    "ablate_4_0_Shuffle3x3_e4",
    "ablate_0_3_Ghost3x3_r2_nose",
    "ablate_0_3_Ghost5x5_r2_nose",
    "ablate_0_3_Ghost7x7_r2_nose",
    "ablate_combo_b1Ghost5x5_b3Shufflee4",
    "ablate_combo_b1Ghost5x5_b14Shufflee2",
    "ablate_combo_b1Ghost5x5_b14Shufflee4",
    "ablate_combo_b1Ghost5x5_b14Shufflee8",
    "ablate_combo_b3Shufflee4_b14Shufflee2",
    "ablate_combo_b3Shufflee4_b14Shufflee4",
    "ablate_combo_b3Shufflee4_b14Shufflee8",
    "ablate_combo_b1Ghost5x5_b3Shufflee4_b14Shufflee4",
    "ablate_pool_0410_014251",
    "ablate_pool_0427_000438",
    "ablate_pool_0502_001503",
    "ablate_pool_0408_233648",
]

KNOWN_RUNNING = {
    "ablate_0_1_Ghost5x5_irstd1k_rerun_g0": "ablate_0_1_Ghost5x5_r2_nose",
    "ablate_0_1_Ghost5x5_irstd1k_rerun_g1": "ablate_0_1_Ghost5x5_r2_nose",
    "ablate_0_1_Ghost5x5_irstd1k_rerun_g4": "ablate_0_1_Ghost5x5_r2_nose",
    "ablate_0_1_Ghost5x5_irstd1k_rerun_g5": "ablate_0_1_Ghost5x5_r2_nose",
    "ablate_0_1_Ghost5x5_irstd1k_rerun_g7": "ablate_0_1_Ghost5x5_r2_nose",
    "ablate_0_3_Shuffle3x3_e4_irstd1k_rerun_g2": "ablate_0_3_Shuffle3x3_e4",
    "ablate_4_0_Shuffle3x3_e8_irstd1k_rerun_g3": "ablate_4_0_Shuffle3x3_e8",
    "ablate_4_0_Shuffle3x3_e2_irstd1k_rerun_g6": "ablate_4_0_Shuffle3x3_e2",
}

VALID_RE = re.compile(
    r"Valid \[(\d+)/1500\]\s+loss\s+([0-9.]+)\s+Validate_IoU\s+([0-9.]+) \(([0-9.]+)\)"
)


def run(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def tmux_sessions():
    proc = run(["tmux", "ls"])
    if proc.returncode != 0:
        return set()
    sessions = set()
    for line in proc.stdout.splitlines():
        if ":" in line:
            sessions.add(line.split(":", 1)[0])
    return sessions


def gpu_memory():
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    if proc.returncode != 0:
        return {}
    mem = {}
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            try:
                mem[int(parts[0])] = int(parts[1])
            except ValueError:
                pass
    return mem


def load_state():
    if STATE_PATH.exists():
        state = json.load(open(STATE_PATH))
    else:
        state = {"launched": [], "queue": []}
    state.setdefault("launched", [])
    state.setdefault("queue", [])
    known = set(state["queue"])
    known.update(entry.get("variant") for entry in state["launched"])
    for variant in QUEUE:
        if variant not in known:
            state["queue"].append(variant)
            known.add(variant)
    return state


def save_state(state):
    tmp = STATE_PATH.with_suffix(".tmp")
    json.dump(state, open(tmp, "w"), indent=2)
    tmp.replace(STATE_PATH)


def parse_logs():
    rows = []
    for path in sorted(Path("/tmp").glob("ablate_*.irstd1k*.log")):
        entries = 0
        last = None
        best = (-1.0, None)
        for line in path.read_text(errors="ignore").splitlines():
            match = VALID_RE.search(line)
            if not match:
                continue
            entries += 1
            epoch = int(match.group(1))
            current = float(match.group(3))
            reported_best = float(match.group(4))
            last = (epoch, current, reported_best)
            if reported_best > best[0]:
                best = (reported_best, epoch)
        if entries:
            rows.append(
                {
                    "log": str(path),
                    "entries": entries,
                    "last": last,
                    "best": best,
                    "hit": best[0] > TARGET,
                }
            )
    rows.sort(key=lambda row: row["best"][0], reverse=True)
    return rows


def write_status(rows, sessions, mem, state):
    lines = [
        f"updated: {datetime.now().isoformat(timespec='seconds')}",
        f"target: Validate_IoU > {TARGET}",
        f"active_sessions: {len(sessions)}",
        f"gpu_memory_mib: {dict(sorted(mem.items()))}",
        f"queued_remaining: {state.get('queue', [])}",
        "",
        "top_runs:",
    ]
    for row in rows[:20]:
        lines.append(
            f"{Path(row['log']).name:70s} entries={row['entries']:4d} "
            f"last={row['last']} best={row['best'][0]:.4f}@{row['best'][1]} hit={row['hit']}"
        )
    STATUS_PATH.write_text("\n".join(lines) + "\n")


def session_name(variant, gpu):
    short = variant.replace("ablate_", "ab_").replace("_r2_nose", "").replace("_", "")
    return f"{short[:70]}_irstd_g{gpu}"


def session_gpu(session):
    match = re.search(r"_g(\d+)$", session)
    if match:
        return int(match.group(1))
    return None


def launch_variant(variant, gpu, state):
    session = session_name(variant, gpu)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"/tmp/{variant}.irstd1k.queue.{stamp}.gpu{gpu}.log"
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} RETRAIN_LOG_PATH={variant} "
        f"bash {SCRIPT} > {log_path} 2>&1"
    )
    proc = run(["tmux", "new-session", "-d", "-s", session, f"bash -lc '{cmd}'"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    state["launched"].append(
        {
            "variant": variant,
            "gpu": gpu,
            "session": session,
            "log": log_path,
            "time": datetime.now().isoformat(timespec="seconds"),
        }
    )
    return session, log_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--free-mem-threshold", type=int, default=1000)
    args = parser.parse_args()

    if not SCRIPT.exists():
        raise SystemExit(f"missing launcher: {SCRIPT}")

    while True:
        state = load_state()
        sessions = tmux_sessions()
        mem = gpu_memory()
        rows = parse_logs()
        write_status(rows, sessions, mem, state)

        hit = next((row for row in rows if row["hit"]), None)
        if hit:
            HIT_PATH.write_text(
                f"hit at {datetime.now().isoformat(timespec='seconds')}\n"
                f"log: {hit['log']}\n"
                f"best: {hit['best'][0]:.4f} @ epoch {hit['best'][1]}\n"
                f"last: {hit['last']}\n"
            )
            save_state(state)
            time.sleep(args.interval)
            continue

        active_variants = {
            variant for session, variant in KNOWN_RUNNING.items() if session in sessions
        }
        active_variants.update(
            entry["variant"]
            for entry in state.get("launched", [])
            if entry.get("session") in sessions
        )

        active_gpus = {
            gpu
            for gpu in (session_gpu(session) for session in KNOWN_RUNNING if session in sessions)
            if gpu is not None
        }
        active_gpus.update(
            int(entry["gpu"])
            for entry in state.get("launched", [])
            if entry.get("session") in sessions and entry.get("gpu") is not None
        )
        free_gpus = [
            gpu
            for gpu, used in sorted(mem.items())
            if used <= args.free_mem_threshold and gpu not in active_gpus
        ]
        while free_gpus and state["queue"]:
            variant = state["queue"].pop(0)
            if variant in active_variants:
                continue
            gpu = free_gpus.pop(0)
            session, log_path = launch_variant(variant, gpu, state)
            active_variants.add(variant)
            STATUS_PATH.write_text(
                STATUS_PATH.read_text()
                + f"\nlaunched: {variant} on gpu {gpu} session={session} log={log_path}\n"
            )

        save_state(state)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
