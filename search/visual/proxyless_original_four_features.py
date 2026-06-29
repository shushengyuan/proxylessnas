#!/usr/bin/env python3
"""Save four ProxylessNAS feature maps with the original visual_mask style."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.utils as tv_utils


SEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEARCH_ROOT.parent
RUN_DIR = SEARCH_ROOT / "logs" / "ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10"
GENE_PATH = (
    SEARCH_ROOT.parent
    / "search1yhy"
    / "0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new"
    / "phase1_gene.txt"
)
DEFAULT_IMAGE = SEARCH_ROOT / "visual" / "visualization_original.png"

# Original script writes L1..L4. These four nodes are the first four diagonal
# nodes from the 5-iteration ProxylessNAS decoder path.
FEATURE_LAYERS = [
    ("L1", 4, 0),
    ("L2", 3, 1),
    ("L3", 2, 2),
    ("L4", 1, 3),
]


def setup_imports() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/proxylessnas_matplotlib")
    for path in (REPO_ROOT, SEARCH_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_proxyless_class():
    for name, path in [
        ("search.models", SEARCH_ROOT / "models"),
        ("search.models.normal_nets", SEARCH_ROOT / "models" / "normal_nets"),
        ("search.models.super_nets", SEARCH_ROOT / "models" / "super_nets"),
    ]:
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package

    proxyless = load_module(
        "search.models.normal_nets.proxyless_nets",
        SEARCH_ROOT / "models" / "normal_nets" / "proxyless_nets.py",
    )
    load_module(
        "search.models.super_nets.super_proxyless_SIRST",
        SEARCH_ROOT / "models" / "super_nets" / "super_proxyless_SIRST.py",
    )
    return proxyless.ProxylessNASNets


def encoder_index(layer: int, iteration: int, total_iterations: int) -> int:
    return sum(total_iterations - i for i in range(iteration)) + layer


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_state(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.343, 0.343, 0.343], [0.231, 0.231, 0.231]),
        ]
    )
    image = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
    return transform(image).unsqueeze(0).to(device)


def original_style_feature(feature: torch.Tensor, output_size: int) -> np.ndarray:
    _, _, width, height = feature.size()
    feature = feature.sum(dim=1).view(1, 1, width, height)
    up_factor = output_size / width
    feature = F.interpolate(feature, scale_factor=up_factor, mode="bicubic", align_corners=False)
    attn = tv_utils.make_grid(feature, nrow=1, normalize=True, scale_each=True)
    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    return cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--gene", type=Path, default=GENE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    setup_imports()
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    net_config = run_dir / "net.config"
    checkpoint = run_dir / "checkpoint" / "model_best.pth.tar"
    output_dir = (args.output_dir or run_dir / "feature_original_four").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(net_config)
    total_iterations = int(config.get("iterations", 5))
    net_class = load_proxyless_class()
    model = net_class.build_from_config(config, str(args.gene.expanduser().resolve()))
    model.load_state_dict(load_checkpoint_state(checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()

    features: dict[str, torch.Tensor] = {}
    handles = []
    for name, layer, iteration in FEATURE_LAYERS:
        idx = encoder_index(layer, iteration, total_iterations)
        handles.append(
            model.encoders[idx].register_forward_hook(
                lambda _module, _inputs, output, name=name: features.__setitem__(name, output.detach())
            )
        )

    for image_path in iter_images(args.image.expanduser().resolve()):
        features.clear()
        image = load_image_tensor(image_path, device)
        print(tuple(image.shape))
        with torch.no_grad():
            model(image)

        for name, layer, iteration in FEATURE_LAYERS:
            feature = features[name]
            attn = original_style_feature(feature, args.output_size)
            output_path = output_dir / f"{image_path.stem}_{name}.png"
            Image.fromarray(np.uint8(attn)).save(output_path)
            print(f"{name}=L({layer},{iteration}) shape={tuple(feature.shape)} attn={attn.shape} -> {output_path}")

    for handle in handles:
        handle.remove()

    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
