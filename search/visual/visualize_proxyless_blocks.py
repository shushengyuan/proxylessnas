#!/usr/bin/env python3
"""Visualize every ProxylessNAS block output from a learned net.config."""

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
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F


SEARCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SEARCH_ROOT.parent
DEFAULT_RUN_DIR = SEARCH_ROOT / "logs" / "ablate_combo_b1Ghost5x5_b3Shufflee2_Retrain_10"
DEFAULT_NET_CONFIG = SEARCH_ROOT / "logs" / "ablate_combo_b1Ghost5x5_b3Shufflee2" / "learned_net" / "net.config"
DEFAULT_IMAGE = SEARCH_ROOT / "visual" / "visualization_original.png"
DEFAULT_GENE_CANDIDATES = [
    REPO_ROOT / "search1yhy" / "0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25_new" / "phase1_gene.txt",
    REPO_ROOT / "search1yhy" / "0_all_search_1_NUAA-SIRST_DNANet_23_10_2023_20_06_25" / "phase1_gene.txt",
]

DATASET_STATS = {
    "IRSTD-SIRST": ((0.343, 0.343, 0.343), (0.231, 0.231, 0.231)),
    "NUAA-SIRST": ((0.439, 0.439, 0.439), (0.217, 0.217, 0.217)),
    "NUDT-SIRST": ((0.423, 0.423, 0.423), (0.217, 0.217, 0.217)),
}


def setup_imports() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/proxylessnas_matplotlib")
    for path in (REPO_ROOT, SEARCH_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


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


def read_parameters(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    if not path.is_file():
        return params
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":--" not in line:
                continue
            key, value = line.split(":--", 1)
            params[key.strip()] = value.strip()
    return params


def resolve_checkpoint(run_dir: Path, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()

    latest_txt = run_dir / "checkpoint" / "latest.txt"
    if latest_txt.is_file():
        latest_path = Path(latest_txt.read_text(encoding="utf-8").strip()).expanduser()
        if not latest_path.is_absolute():
            latest_path = (latest_txt.parent / latest_path).resolve()
        if latest_path.is_file():
            return latest_path.resolve()

    for candidate in ("checkpoint.pth.tar", "model_best.pth.tar"):
        path = run_dir / "checkpoint" / candidate
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"no checkpoint found under {run_dir / 'checkpoint'}")


def resolve_gene(run_dir: Path, net_config: Path, gene: Path | None) -> Path:
    if gene is not None:
        return gene.expanduser().resolve()

    candidates: list[Path] = []
    param_paths = [
        run_dir / "parameters.txt",
        run_dir.parent / "parameters.txt",
        net_config.parent / "parameters.txt",
        net_config.parent.parent / "parameters.txt",
    ]
    for param_path in param_paths:
        params = read_parameters(param_path)
        gene_text = params.get("gene")
        if not gene_text:
            continue
        external = Path(gene_text).expanduser()
        if external.is_file():
            candidates.append(external.resolve())
        wanted_parent = external.parent.name.removesuffix("_new")
        search_root = REPO_ROOT / "search1yhy"
        if search_root.is_dir():
            for local_gene in sorted(search_root.glob("*/phase1_gene.txt")):
                if local_gene.parent.name.removesuffix("_new") == wanted_parent:
                    candidates.append(local_gene.resolve())

    candidates.extend(path.resolve() for path in DEFAULT_GENE_CANDIDATES if path.is_file())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("gene file could not be resolved; please pass --gene explicitly")


def load_checkpoint_state(path: Path) -> dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def all_blocks(total_iterations: int) -> list[tuple[int, int, int]]:
    blocks = []
    index = 0
    for iteration in range(total_iterations):
        for layer in range(total_iterations - iteration):
            blocks.append((index, layer, iteration))
            index += 1
    return blocks


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in suffixes)


def load_image(
    path: Path,
    device: torch.device,
    input_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(path).convert("RGB").resize((input_size, input_size), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return image, tensor


def feature_to_heatmap(feature: torch.Tensor, output_size: int, reduce: str) -> Image.Image:
    if reduce == "sum":
        feature = feature.sum(dim=1, keepdim=True)
    elif reduce == "mean":
        feature = feature.mean(dim=1, keepdim=True)
    elif reduce == "abs-sum":
        feature = feature.abs().sum(dim=1, keepdim=True)
    elif reduce == "max":
        feature = feature.max(dim=1, keepdim=True)[0]
    else:
        raise ValueError(f"unsupported reduce mode: {reduce}")
    feature = F.interpolate(feature.float(), size=(output_size, output_size), mode="bicubic", align_corners=False)
    array = feature.squeeze().detach().cpu().numpy()
    array = np.nan_to_num(array)
    low = float(array.min())
    high = float(array.max())
    if high > low:
        array = (array - low) / (high - low)
    else:
        array = np.zeros_like(array, dtype=np.float32)
    gray = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color)


def labeled_tile(label: str, image: Image.Image) -> Image.Image:
    tile = Image.new("RGB", (image.width, image.height + 34), (255, 255, 255))
    tile.paste(image, (0, 34))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 6), label, fill=(0, 0, 0), font=font)
    return tile


def save_grid(items: list[tuple[str, Image.Image]], path: Path, columns: int) -> None:
    if not items:
        return
    tiles = [labeled_tile(label, image) for label, image in items]
    cols = max(1, columns)
    rows = (len(tiles) + cols - 1) // cols
    tile_w = max(tile.width for tile in tiles)
    tile_h = max(tile.height for tile in tiles)
    grid = Image.new("RGB", (tile_w * cols, tile_h * rows), (255, 255, 255))
    for idx, tile in enumerate(tiles):
        x = (idx % cols) * tile_w
        y = (idx // cols) * tile_h
        grid.paste(tile, (x, y))
    grid.save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--net-config", type=Path, default=DEFAULT_NET_CONFIG)
    parser.add_argument("--run-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--gene", type=Path, default=None)
    parser.add_argument("--dataset", choices=sorted(DATASET_STATS), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--reduce", choices=("sum", "mean", "abs-sum", "max"), default="sum")
    parser.add_argument("--grid-columns", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    setup_imports()
    args = parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    net_config = args.net_config.expanduser().resolve()
    run_config = args.run_config.expanduser().resolve() if args.run_config is not None else (
        (net_config.parent / "run.config").resolve() if (net_config.parent / "run.config").is_file() else None
    )
    checkpoint = resolve_checkpoint(run_dir, args.checkpoint)
    gene = resolve_gene(run_dir, net_config, args.gene)
    output_dir = (args.output_dir or run_dir / "feature_all_blocks_visual").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    net_cfg = read_json(net_config)
    run_cfg = read_json(run_config) if run_config is not None else {}
    net_cfg.setdefault("more_down", False)
    total_iterations = int(net_cfg.get("iterations", 5))
    block_specs = all_blocks(total_iterations)
    dataset = args.dataset or run_cfg.get("dataset", "IRSTD-SIRST")
    if dataset not in DATASET_STATS:
        raise ValueError(f"unsupported dataset stats for {dataset!r}")
    mean, std = DATASET_STATS[dataset]
    input_size = args.input_size if args.input_size is not None else int(run_cfg.get("crop_size", 256))

    net_class = load_proxyless_class()
    model = net_class.build_from_config(net_cfg, str(gene))
    model.load_state_dict(load_checkpoint_state(checkpoint))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model.to(device).eval()

    captured: dict[int, torch.Tensor] = {}
    semantic_features: dict[str, torch.Tensor] = {}
    handles = []
    block_cfgs = net_cfg["blocks"]
    if len(block_specs) != len(block_cfgs) or len(model.encoders) != len(block_cfgs):
        raise ValueError(
            "block count mismatch: "
            f"iteration layout={len(block_specs)}, net.config={len(block_cfgs)}, model.encoders={len(model.encoders)}"
        )

    for block_index, _layer, _iteration in block_specs:
        handles.append(
            model.encoders[block_index].register_forward_hook(
                lambda _module, _inputs, output, block_index=block_index: captured.__setitem__(
                    block_index, output.detach()
                )
            )
        )
    handles.append(
        model.post_transform_conv_block.register_forward_pre_hook(
            lambda _module, inputs: semantic_features.__setitem__("head_input", inputs[0].detach())
        )
    )
    handles.append(
        model.post_transform_conv_block.register_forward_hook(
            lambda _module, _inputs, output: semantic_features.__setitem__("head_output", output.detach())
        )
    )

    manifest: list[dict] = []
    for image_path in iter_images(args.image.expanduser().resolve()):
        image_dir = output_dir / image_path.stem
        blocks_dir = image_dir / "blocks"
        semantic_dir = image_dir / "semantic"
        image_dir.mkdir(parents=True, exist_ok=True)
        blocks_dir.mkdir(parents=True, exist_ok=True)
        semantic_dir.mkdir(parents=True, exist_ok=True)
        captured.clear()
        semantic_features.clear()

        input_image, tensor = load_image(image_path, device, input_size, mean, std)
        input_image.save(image_dir / "input.png")

        with torch.no_grad():
            model_output = model(tensor)

        if isinstance(model_output, tuple) and len(model_output) == 2:
            head_output, tail_features = model_output
        else:
            head_output, tail_features = model_output, []
        semantic_features.setdefault("head_output", head_output.detach())

        grid_items: list[tuple[str, Image.Image]] = []
        for block_index, layer, iteration in block_specs:
            feature = captured[block_index]
            block_name = block_cfgs[block_index]["name"]
            skipped = bool(getattr(model.encoders[block_index], "skipped", False))
            label = f"b{block_index:02d}_{block_name}_L{layer}_{iteration}"
            if skipped:
                label += "_skipped"
            heatmap = feature_to_heatmap(feature, args.output_size, args.reduce)
            output_path = blocks_dir / f"{label}.png"
            heatmap.save(output_path)
            display_label = f"b{block_index:02d} {block_name} L({layer},{iteration})"
            if skipped:
                display_label += " SKIP"
            grid_items.append((display_label, heatmap))
            manifest.append(
                {
                    "image": image_path.name,
                    "block_index": block_index,
                    "block_name": block_name,
                    "skipped": skipped,
                    "layer": layer,
                    "iteration": iteration,
                    "shape": list(feature.shape),
                    "file": str(output_path.relative_to(output_dir)),
                }
            )
            print(f"{label} shape={tuple(feature.shape)} -> {output_path}")

        save_grid(grid_items, image_dir / "grid_blocks.png", args.grid_columns)

        semantic_items: list[tuple[str, Image.Image]] = []
        for tail_index, feature in enumerate(tail_features):
            layer = total_iterations - 1 - tail_index
            label = f"s{tail_index:02d}_tail_L{layer}_{tail_index}"
            heatmap = feature_to_heatmap(feature.detach(), args.output_size, args.reduce)
            output_path = semantic_dir / f"{label}.png"
            heatmap.save(output_path)
            semantic_items.append((f"s{tail_index:02d} tail L({layer},{tail_index})", heatmap))
            manifest.append(
                {
                    "image": image_path.name,
                    "group": "semantic_tail",
                    "tail_index": tail_index,
                    "layer": layer,
                    "iteration": tail_index,
                    "shape": list(feature.shape),
                    "file": str(output_path.relative_to(output_dir)),
                }
            )
            print(f"{label} shape={tuple(feature.shape)} -> {output_path}")

        for key in ("head_input", "head_output"):
            feature = semantic_features[key]
            label = f"s{len(semantic_items):02d}_{key}"
            heatmap = feature_to_heatmap(feature, args.output_size, args.reduce)
            output_path = semantic_dir / f"{label}.png"
            heatmap.save(output_path)
            semantic_items.append((key, heatmap))
            manifest.append(
                {
                    "image": image_path.name,
                    "group": "semantic_head",
                    "name": key,
                    "shape": list(feature.shape),
                    "file": str(output_path.relative_to(output_dir)),
                }
            )
            print(f"{label} shape={tuple(feature.shape)} -> {output_path}")

        save_grid(semantic_items, image_dir / "grid_semantic.png", args.grid_columns)
        print(f"{image_path.name} -> {image_dir}")

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for handle in handles:
        handle.remove()

    print(f"net.config: {net_config}")
    print(f"checkpoint: {checkpoint}")
    print(f"gene: {gene}")
    print(f"saved to: {output_dir}")


if __name__ == "__main__":
    main()
