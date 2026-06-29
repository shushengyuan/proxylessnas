#!/usr/bin/env python3
"""Generate heatmap visualizations for selected ProxylessNAS feature nodes."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont


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


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def checkpoint_state(path: Path) -> dict:
    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def layer_to_encoder_index(layer: int, iteration: int, total_iterations: int = 5) -> int:
    return sum(total_iterations - i for i in range(iteration)) + layer


def preprocess(path: Path, size: int):
    import torch

    image = Image.open(path).convert("RGB")
    resized = image.resize((size, size), Image.BILINEAR)
    data = np.asarray(resized, dtype=np.float32) / 255.0
    mean = np.asarray([0.343, 0.343, 0.343], dtype=np.float32)
    std = np.asarray([0.231, 0.231, 0.231], dtype=np.float32)
    data = (data - mean) / std
    tensor = torch.from_numpy(data.transpose(2, 0, 1)).unsqueeze(0)
    return image, resized, tensor


def activation_map(
    feature,
    output_size: int,
    channel_reduce: str,
    power: float,
    percentile: tuple[float, float],
) -> np.ndarray:
    import torch.nn.functional as F

    feature = feature.detach().float().abs()
    if channel_reduce == "max":
        act = feature.max(dim=1, keepdim=True)[0]
    elif channel_reduce == "mean":
        act = feature.mean(dim=1, keepdim=True)
    elif channel_reduce == "sum":
        act = feature.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"unsupported channel_reduce: {channel_reduce}")

    if power != 1.0:
        act = act.clamp_min(0) ** power
    act = act.cpu()
    act = F.interpolate(act, size=(output_size, output_size), mode="bicubic", align_corners=False)
    array = act.squeeze().numpy()
    array = np.nan_to_num(array)
    low, high = np.percentile(array, percentile)
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    array = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (array * 255).astype(np.uint8)


def heatmap(gray: np.ndarray, colormap: str) -> Image.Image:
    import cv2

    colormaps = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    color = cv2.applyColorMap(gray, colormaps[colormap])
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color)


def overlay(base: Image.Image, heat: Image.Image, alpha: float) -> Image.Image:
    base = base.resize(heat.size, Image.BILINEAR).convert("RGB")
    return Image.blend(base, heat.convert("RGB"), alpha)


def labeled_tile(label: str, image: Image.Image) -> Image.Image:
    tile = Image.new("RGB", (image.width, image.height + 32), (255, 255, 255))
    tile.paste(image.convert("RGB"), (0, 32))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 5), label, fill=(0, 0, 0), font=font)
    return tile


def save_grid(items: list[tuple[str, Image.Image]], path: Path) -> None:
    tiles = [labeled_tile(label, image) for label, image in items]
    grid = Image.new("RGB", (sum(t.width for t in tiles), max(t.height for t in tiles)), (255, 255, 255))
    x = 0
    for tile in tiles:
        grid.paste(tile, (x, 0))
        x += tile.width
    grid.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR)
    parser.add_argument("--gene", type=Path, default=GENE_PATH)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--channel-reduce", choices=["max", "mean", "sum"], default="max")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--percentile", type=float, nargs=2, default=(5.0, 99.5))
    parser.add_argument("--colormap", choices=["jet", "turbo", "inferno"], default="jet")
    return parser.parse_args()


def main() -> None:
    setup_imports()
    args = parse_args()

    import torch

    run_dir = args.run_dir.expanduser().resolve()
    net_config = run_dir / "net.config"
    checkpoint = run_dir / "checkpoint" / "model_best.pth.tar"
    output_dir = (args.output_dir or run_dir / "feature_heatmap_standalone").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(net_config)
    net_class = load_proxyless_class()
    model = net_class.build_from_config(config, str(args.gene.expanduser().resolve()))
    model.load_state_dict(checkpoint_state(checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()

    requested = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4)]
    captured = {}
    handles = []
    for layer, iteration in requested:
        label = f"L({layer},{iteration})"
        index = layer_to_encoder_index(layer, iteration, int(config.get("iterations", 5)))
        handles.append(
            model.encoders[index].register_forward_hook(
                lambda _m, _i, output, label=label: captured.__setitem__(label, output)
            )
        )

    original, resized, tensor = preprocess(args.image.expanduser().resolve(), args.input_size)
    with torch.no_grad():
        model(tensor.to(device))

    heat_tiles = []
    overlay_tiles = []
    for layer, iteration in requested:
        label = f"L({layer},{iteration})"
        gray = activation_map(
            captured[label],
            args.output_size,
            args.channel_reduce,
            args.power,
            tuple(args.percentile),
        )
        heat = heatmap(gray, args.colormap)
        blended = overlay(original, heat, args.alpha)

        stem = args.image.stem
        safe_label = f"L{layer}_{iteration}"
        heat_path = output_dir / f"{stem}_{safe_label}_heatmap.png"
        overlay_path = output_dir / f"{stem}_{safe_label}_overlay.png"
        heat.save(heat_path)
        blended.save(overlay_path)

        print(f"{label}: {tuple(captured[label].shape)} -> {heat_path}")
        heat_tiles.append((label, heat))
        overlay_tiles.append((label, blended))

    save_grid(heat_tiles, output_dir / f"{args.image.stem}_heatmap_grid.png")
    save_grid(overlay_tiles, output_dir / f"{args.image.stem}_overlay_grid.png")

    for handle in handles:
        handle.remove()

    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
