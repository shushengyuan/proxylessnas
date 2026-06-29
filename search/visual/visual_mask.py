#!/usr/bin/env python3
"""Visualize ProxylessNAS SIRST intermediate feature maps."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEARCH_ROOT.parent
DEFAULT_RUN_DIR = SEARCH_ROOT / "logs" / "ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10"
DEFAULT_GENE = (
    SEARCH_ROOT.parent
    / "search1yhy"
    / "0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new"
    / "phase1_gene.txt"
)
DEFAULT_IMAGE = SEARCH_ROOT / "visual" / "visualization_original.png"

DATASET_STATS = {
    "IRSTD-SIRST": ((0.343, 0.343, 0.343), (0.231, 0.231, 0.231)),
    "NUAA-SIRST": ((0.439, 0.439, 0.439), (0.217, 0.217, 0.217)),
    "NUAA-SIRST-Old": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def setup_paths() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/proxylessnas_matplotlib")
    for path in (REPO_ROOT, SEARCH_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_proxyless_builder():
    # Avoid executing search/models/__init__.py; it imports run_manager and is
    # unnecessary for standalone feature visualization.
    for name, path in [
        ("search.models", SEARCH_ROOT / "models"),
        ("search.models.normal_nets", SEARCH_ROOT / "models" / "normal_nets"),
        ("search.models.super_nets", SEARCH_ROOT / "models" / "super_nets"),
    ]:
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [str(path)]
            sys.modules[name] = pkg

    proxyless_mod = load_module(
        "search.models.normal_nets.proxyless_nets",
        SEARCH_ROOT / "models" / "normal_nets" / "proxyless_nets.py",
    )
    load_module(
        "search.models.super_nets.super_proxyless_SIRST",
        SEARCH_ROOT / "models" / "super_nets" / "super_proxyless_SIRST.py",
    )
    return proxyless_mod.ProxylessNASNets


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def torch_load(path: Path, map_location):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def strip_module_prefix(state_dict: dict) -> dict:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def parse_layer_spec(spec: str, iterations: int) -> list[tuple[str, int, int]]:
    """Return [(label, layer, iteration), ...] for strings like L(4,0),L(3,1)."""
    if spec == "diagonal":
        return [(f"L({iterations - 1 - i},{i})", iterations - 1 - i, i) for i in range(iterations)]

    layers: list[tuple[str, int, int]] = []
    for item in spec.split(","):
        item = item.strip()
        match = re.fullmatch(r"L\((\d+),(\d+)\)", item)
        if not match:
            raise ValueError(f"invalid layer spec {item!r}; use L(layer,iteration)")
        layer = int(match.group(1))
        iteration = int(match.group(2))
        layers.append((f"L({layer},{iteration})", layer, iteration))
    return layers


def encoder_index(layer: int, iteration: int, iterations: int) -> int:
    if iteration < 0 or iteration >= iterations:
        raise ValueError(f"iteration out of range for L({layer},{iteration})")
    if layer < 0 or layer >= iterations - iteration:
        raise ValueError(f"layer out of range for L({layer},{iteration})")
    return sum(iterations - idx for idx in range(iteration)) + layer


def resolve_input_size(run_dir: Path, input_size: int | None) -> int:
    if input_size is not None:
        return input_size
    run_config = run_dir / "run.config"
    if run_config.is_file():
        return int(read_json(run_config).get("crop_size", 256))
    return 256


def resolve_stats(run_dir: Path, dataset: str | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if dataset:
        return DATASET_STATS[dataset]
    run_config = run_dir / "run.config"
    if run_config.is_file():
        run_dataset = read_json(run_config).get("dataset")
        if run_dataset in DATASET_STATS:
            return DATASET_STATS[run_dataset]
    return DATASET_STATS["IRSTD-SIRST"]


def preprocess_image(path: Path, input_size: int, mean: tuple[float, ...], std: tuple[float, ...]):
    import torch

    image = Image.open(path).convert("RGB")
    resized = image.resize((input_size, input_size), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return image, resized, tensor


def reduce_feature(feature, reduce: str):
    if reduce == "sum":
        return feature.sum(dim=1, keepdim=True)
    if reduce == "mean":
        return feature.mean(dim=1, keepdim=True)
    if reduce == "max":
        return feature.max(dim=1, keepdim=True).values
    if reduce == "absmean":
        return feature.abs().mean(dim=1, keepdim=True)
    raise ValueError(f"unknown reduce mode: {reduce}")


def feature_to_uint8(feature, output_size: int, reduce: str) -> np.ndarray:
    import torch.nn.functional as F

    activation = reduce_feature(feature.float().cpu(), reduce)
    activation = F.interpolate(
        activation,
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=False,
    )
    array = activation.squeeze().numpy()
    array = np.nan_to_num(array)
    low = float(array.min())
    high = float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)
    array = (array - low) / (high - low)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def apply_colormap(gray: np.ndarray, colormap: str) -> Image.Image:
    if colormap == "gray":
        return Image.fromarray(gray).convert("RGB")

    import cv2

    cv_maps = {
        "jet": cv2.COLORMAP_JET,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    if colormap not in cv_maps:
        raise ValueError(f"unsupported colormap: {colormap}")
    bgr = cv2.applyColorMap(gray, cv_maps[colormap])
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def label_image(image: Image.Image, label: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 28), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 4), label, fill=(0, 0, 0), font=font)
    return canvas


def save_grid(images: list[tuple[str, Image.Image]], output_path: Path) -> None:
    if not images:
        return
    labeled = [label_image(image, label) for label, image in images]
    width = sum(image.width for image in labeled)
    height = max(image.height for image in labeled)
    grid = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for image in labeled:
        grid.paste(image, (x, 0))
        x += image.width
    grid.save(output_path)


def iter_image_paths(image_path: Path) -> list[Path]:
    if image_path.is_file():
        return [image_path]
    if image_path.is_dir():
        suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return sorted(path for path in image_path.iterdir() if path.suffix.lower() in suffixes)
    raise FileNotFoundError(f"image path not found: {image_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE, help="Image file or image directory.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Retrain log directory.")
    parser.add_argument("--net-config", type=Path, default=None, help="Defaults to RUN_DIR/net.config.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Defaults to RUN_DIR/checkpoint/model_best.pth.tar.")
    parser.add_argument("--gene", type=Path, default=DEFAULT_GENE, help="phase1_gene.txt used by this architecture.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for generated feature maps.")
    parser.add_argument("--layers", default="diagonal", help="Layer list, e.g. L(4,0),L(3,1),L(2,2),L(1,3),L(0,4).")
    parser.add_argument("--dataset", choices=sorted(DATASET_STATS), default=None, help="Normalization stats override.")
    parser.add_argument("--input-size", type=int, default=None, help="Inference resize size. Defaults to run.config crop_size.")
    parser.add_argument("--output-size", type=int, default=512, help="Feature-map output size.")
    parser.add_argument("--reduce", choices=["sum", "mean", "max", "absmean"], default="sum")
    parser.add_argument("--colormap", choices=["gray", "jet", "turbo", "inferno"], default="gray")
    parser.add_argument("--device", default="cuda:0", help="Use cuda:0 or cpu.")
    parser.add_argument("--save-grid", action="store_true", help="Also save one horizontal grid image per input.")
    return parser.parse_args()


def main() -> None:
    setup_paths()
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    net_config = (args.net_config or run_dir / "net.config").expanduser().resolve()
    checkpoint = (args.checkpoint or run_dir / "checkpoint" / "model_best.pth.tar").expanduser().resolve()
    gene = args.gene.expanduser().resolve()
    output_dir = (args.output_dir or run_dir / "feature_maps").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not net_config.is_file():
        raise FileNotFoundError(f"net.config not found: {net_config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not gene.is_file():
        raise FileNotFoundError(f"gene file not found: {gene}")

    import torch

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    ProxylessNASNets = load_proxyless_builder()
    net_cfg = read_json(net_config)
    net = ProxylessNASNets.build_from_config(net_cfg, str(gene))
    loaded = torch_load(checkpoint, map_location="cpu")
    state_dict = strip_module_prefix(loaded.get("state_dict", loaded))
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    iterations = int(net_cfg.get("iterations", 5))
    requested_layers = parse_layer_spec(args.layers, iterations)
    target_indices = {
        label: encoder_index(layer, iteration, iterations)
        for label, layer, iteration in requested_layers
    }

    features = {}
    handles = []
    for label, idx in target_indices.items():
        handle = net.encoders[idx].register_forward_hook(
            lambda _module, _inputs, output, label=label: features.__setitem__(label, output.detach())
        )
        handles.append(handle)

    input_size = resolve_input_size(run_dir, args.input_size)
    mean, std = resolve_stats(run_dir, args.dataset)
    image_paths = iter_image_paths(args.image.expanduser().resolve())

    with torch.no_grad():
        for image_path in image_paths:
            features.clear()
            _, _, tensor = preprocess_image(image_path, input_size, mean, std)
            net(tensor.to(device))

            grid_images: list[tuple[str, Image.Image]] = []
            for label, _layer, _iteration in requested_layers:
                feature = features.get(label)
                if feature is None:
                    raise RuntimeError(f"feature was not captured for {label}")
                gray = feature_to_uint8(feature, args.output_size, args.reduce)
                feature_image = apply_colormap(gray, args.colormap)
                safe_label = label.replace("(", "").replace(")", "").replace(",", "_")
                output_path = output_dir / f"{image_path.stem}_{safe_label}.png"
                feature_image.save(output_path)
                grid_images.append((label, feature_image))
                print(f"{label}: shape={tuple(feature.shape)} -> {output_path}")

            if args.save_grid:
                save_grid(grid_images, output_dir / f"{image_path.stem}_feature_grid.png")

    for handle in handles:
        handle.remove()

    print(f"saved feature maps to: {output_dir}")


if __name__ == "__main__":
    main()
