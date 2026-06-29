#!/usr/bin/env python3
"""Create CPU-base Ghost+Shuffle ablation variants for IRSTD retraining."""

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
CPU_NET = ROOT / "cpu_net.config"
BASE_RUN = LOGS / "0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00" / "learned_net" / "run.config"


def ghost5(old):
    return {
        "name": "GhostConvLayer",
        "in_channels": old["in_channels"],
        "out_channels": old["out_channels"],
        "kernel_size": 5,
        "stride": old.get("stride", 1),
        "num_blocks": old.get("num_blocks", 1),
        "expand_ratio": 2,
        "mid_channels": None,
        "ghost_ratio": 2,
        "se_ratio": 0.0,
    }


def shuffle(old, expand_ratio):
    return {
        "name": "ShuffleConvLayer",
        "in_channels": old["in_channels"],
        "out_channels": old["out_channels"],
        "kernel_size": 3,
        "stride": old.get("stride", 1),
        "num_blocks": old.get("num_blocks", 1),
        "expand_ratio": expand_ratio,
        "shuffle_groups": 2,
    }


VARIANTS = {
    "ablate_cpu_combo_b1Ghost5x5_b3Shufflee2": [
        (1, "ghost5", ghost5),
        (3, "shuffle2", lambda old: shuffle(old, 2)),
    ],
    "ablate_cpu_combo_b3Shufflee2_b8Ghost5x5": [
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (8, "ghost5", ghost5),
    ],
    "ablate_cpu_combo_b1Ghost5x5_b2Shufflee8": [
        (1, "ghost5", ghost5),
        (2, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_cpu_combo_b4Ghost5x5_b3Shufflee2": [
        (4, "ghost5", ghost5),
        (3, "shuffle2", lambda old: shuffle(old, 2)),
    ],
    # Avoid the GPU-best positions (b1 + b3) while keeping one Ghost and one Shuffle.
    "ablate_cpu_combo_b2Ghost5x5_b8Shufflee4": [
        (2, "ghost5", ghost5),
        (8, "shuffle4", lambda old: shuffle(old, 4)),
    ],
    "ablate_cpu_combo_b2Shufflee8_b8Ghost5x5": [
        (2, "shuffle8", lambda old: shuffle(old, 8)),
        (8, "ghost5", ghost5),
    ],
    "ablate_cpu_combo_b2Shufflee4_b8Ghost5x5": [
        (2, "shuffle4", lambda old: shuffle(old, 4)),
        (8, "ghost5", ghost5),
    ],
    "ablate_cpu_combo_b2Shufflee2_b8Ghost5x5": [
        (2, "shuffle2", lambda old: shuffle(old, 2)),
        (8, "ghost5", ghost5),
    ],
    "ablate_cpu_combo_b4Ghost5x5_b2Shufflee8": [
        (4, "ghost5", ghost5),
        (2, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_cpu_combo_b2Ghost5x5_b4Shufflee8": [
        (2, "ghost5", ghost5),
        (4, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_cpu_combo_b2Ghost5x5_b4Shufflee4": [
        (2, "ghost5", ghost5),
        (4, "shuffle4", lambda old: shuffle(old, 4)),
    ],
    "ablate_cpu_combo_b2Ghost5x5_b4Shufflee2": [
        (2, "ghost5", ghost5),
        (4, "shuffle2", lambda old: shuffle(old, 2)),
    ],
    "ablate_cpu_combo_b8Ghost5x5_b4Shufflee2": [
        (8, "ghost5", ghost5),
        (4, "shuffle2", lambda old: shuffle(old, 2)),
    ],
}


def create_variant(name, changes):
    dst = LOGS / name
    learned = dst / "learned_net"
    learned.mkdir(parents=True, exist_ok=True)

    net = json.load(open(CPU_NET))
    applied = []
    for block_idx, kind, make_config in changes:
        old = copy.deepcopy(net["blocks"][block_idx])
        new = make_config(old)
        net["blocks"][block_idx] = new
        applied.append(
            {
                "block": block_idx,
                "kind": kind,
                "old_config": old,
                "new_config": new,
            }
        )

    json.dump(net, open(learned / "net.config", "w"), indent=4)
    json.dump(json.load(open(BASE_RUN)), open(learned / "run.config", "w"), indent=4)
    json.dump(
        {
            "base_net": str(CPU_NET),
            "base_run": str(BASE_RUN),
            "name": name,
            "desc": "CPU-base targeted Ghost+Shuffle replacement ablation",
            "changes": applied,
        },
        open(dst / "ablation.json", "w"),
        indent=4,
    )
    print(name)


def main():
    if not CPU_NET.exists():
        raise SystemExit(f"missing CPU net config: {CPU_NET}")
    if not BASE_RUN.exists():
        raise SystemExit(f"missing base run config: {BASE_RUN}")
    for name, changes in VARIANTS.items():
        create_variant(name, changes)


if __name__ == "__main__":
    main()
