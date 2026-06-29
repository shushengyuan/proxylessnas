#!/usr/bin/env python3
"""Create targeted ablation variants around the strongest IRSTD-1K runs."""

import copy
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
BASE = LOGS / "0,1_NUAA-SIRST_Super_all_Res_Group_Spa_MBConv_27_10_2023_10_47_00"


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
    "ablate_combo_b1Ghost5x5_b3Shufflee2": [
        (1, "ghost5", ghost5),
        (3, "shuffle2", lambda old: shuffle(old, 2)),
    ],
    "ablate_combo_b3Shufflee2_b14Shufflee8": [
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (14, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_combo_b1Ghost5x5_b3Shufflee2_b14Shufflee8": [
        (1, "ghost5", ghost5),
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (14, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_combo_b1Ghost5x5_b2Ghost5x5": [
        (1, "ghost5", ghost5),
        (2, "ghost5", ghost5),
    ],
    "ablate_combo_b1Ghost5x5_b2Shufflee8": [
        (1, "ghost5", ghost5),
        (2, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_combo_b3Shufflee2_b8Ghost5x5": [
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (8, "ghost5", ghost5),
    ],
    "ablate_combo_b1Ghost5x5_b3Shufflee2_b8Ghost5x5": [
        (1, "ghost5", ghost5),
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (8, "ghost5", ghost5),
    ],
    "ablate_combo_b3Shufflee2_b14Shufflee2": [
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (14, "shuffle2", lambda old: shuffle(old, 2)),
    ],
    "ablate_combo_b1Ghost5x5_b3Shufflee2_b14Shufflee2": [
        (1, "ghost5", ghost5),
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (14, "shuffle2", lambda old: shuffle(old, 2)),
    ],
    "ablate_combo_b3Shufflee2_b2Shufflee8": [
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (2, "shuffle8", lambda old: shuffle(old, 8)),
    ],
    "ablate_combo_b3Shufflee2_b8Shufflee4": [
        (3, "shuffle2", lambda old: shuffle(old, 2)),
        (8, "shuffle4", lambda old: shuffle(old, 4)),
    ],
    "ablate_combo_b1Ghost5x5_b8Ghost5x5": [
        (1, "ghost5", ghost5),
        (8, "ghost5", ghost5),
    ],
}


def copy_base(dst):
    if dst.exists():
        return
    shutil.copytree(BASE, dst)


def create_variant(name, changes):
    dst = LOGS / name
    copy_base(dst)

    net_path = dst / "learned_net" / "net.config"
    net = json.load(open(net_path))
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

    json.dump(net, open(net_path, "w"), indent=4)
    json.dump(
        {
            "base": str(BASE),
            "name": name,
            "desc": "targeted combined replacement ablation",
            "changes": applied,
        },
        open(dst / "ablation.json", "w"),
        indent=4,
    )
    print(name)


def main():
    if not BASE.exists():
        raise SystemExit(f"missing base log: {BASE}")
    for name, changes in VARIANTS.items():
        create_variant(name, changes)


if __name__ == "__main__":
    main()
