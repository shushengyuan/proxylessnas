#!/usr/bin/env python3
"""DARTS genotype adapted to the SIRST binary segmentation pipeline."""

from __future__ import annotations

from collections import namedtuple
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


Genotype = namedtuple("Genotype", "normal normal_concat reduce reduce_concat")


def _count_leaf_flops(module: nn.Module, x: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    y = module(x)
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        params = float(module.weight.numel())
        flops = params * float(y.size(2) * y.size(3))
        return flops, params, y
    if isinstance(module, nn.Linear):
        params = float(module.weight.numel())
        flops = params * float(x.size(0))
        return flops, params, y
    return 0.0, 0.0, y


def _count_sequential_flops(module: nn.Sequential, x: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    total_flops = 0.0
    total_params = 0.0
    for sub_module in module:
        if hasattr(sub_module, "get_flops"):
            delta_flops, delta_params, x = sub_module.get_flops(x)
        else:
            delta_flops, delta_params, x = _count_leaf_flops(sub_module, x)
        total_flops += float(delta_flops)
        total_params += float(delta_params)
    return total_flops, total_params, x


def _count_executed_module_flops(module: nn.Module, *inputs: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    """Count Conv/Linear work from the module's actual forward path."""

    total_flops = 0.0
    total_params = 0.0
    handles = []

    def hook(sub_module: nn.Module, args, output):
        nonlocal total_flops, total_params
        x = args[0]
        if isinstance(sub_module, (nn.Conv2d, nn.ConvTranspose2d)):
            params = float(sub_module.weight.numel())
            total_params += params
            total_flops += params * float(output.size(2) * output.size(3))
        elif isinstance(sub_module, nn.Linear):
            params = float(sub_module.weight.numel())
            total_params += params
            total_flops += params * float(x.size(0))

    for sub_module in module.modules():
        if isinstance(sub_module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(sub_module.register_forward_hook(hook))

    try:
        output = module(*inputs)
    finally:
        for handle in handles:
            handle.remove()
    return total_flops, total_params, output


def genotype_from_str(value: str) -> Genotype:
    """Parse the DARTS string representation saved by quark0/darts."""

    return eval(value, {"Genotype": Genotype, "range": range})


class DropPath_(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.0:
            keep_prob = 1.0 - self.p
            mask = x.new_empty(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
            x = x.div(keep_prob).mul(mask)
        return x

    def get_flops(self, x):
        return 0.0, 0.0, self.forward(x)


class PoolBN(nn.Module):
    def __init__(self, pool_type: str, channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        if pool_type == "max":
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type == "avg":
            self.pool = nn.AvgPool2d(kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError(f"unsupported pool type: {pool_type}")
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        return self.bn(self.pool(x))

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


class StdConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.net(x)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


class DilConv(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        return self.net(x)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


class SepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.net = nn.Sequential(
            DilConv(in_ch, in_ch, kernel_size, stride, padding, dilation=1),
            DilConv(in_ch, out_ch, kernel_size, 1, padding, dilation=1),
        )

    def forward(self, x):
        return self.net(x)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


class Identity(nn.Module):
    def forward(self, x):
        return x

    def get_flops(self, x):
        return 0.0, 0.0, x


class Zero(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x * 0.0
        return x[:, :, :: self.stride, :: self.stride] * 0.0

    def get_flops(self, x):
        return 0.0, 0.0, self.forward(x)


class FactorizedReduce(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        if out_ch % 2 != 0:
            raise ValueError("FactorizedReduce requires an even output channel count")
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(out)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


def build_op(op_name: str, channels: int, stride: int) -> nn.Module:
    ops = {
        "none": lambda: Zero(stride),
        "avg_pool_3x3": lambda: PoolBN("avg", channels, 3, stride, 1),
        "max_pool_3x3": lambda: PoolBN("max", channels, 3, stride, 1),
        "skip_connect": lambda: Identity() if stride == 1 else FactorizedReduce(channels, channels),
        "sep_conv_3x3": lambda: SepConv(channels, channels, 3, stride, 1),
        "sep_conv_5x5": lambda: SepConv(channels, channels, 5, stride, 2),
        "dil_conv_3x3": lambda: DilConv(channels, channels, 3, stride, 2, dilation=2),
        "dil_conv_5x5": lambda: DilConv(channels, channels, 5, stride, 4, dilation=2),
    }
    if op_name not in ops:
        raise ValueError(f"unsupported DARTS op: {op_name}")
    return ops[op_name]()


class DartsCell(nn.Module):
    """Discrete DARTS cell, using the normal genotype inside the segmentation skeleton."""

    def __init__(self, genotype: Genotype, in_ch: int, out_ch: int):
        super().__init__()
        cell_ch = max(1, out_ch // 4)
        self.preproc0 = StdConv(in_ch, cell_ch, 1, 1, 0)
        self.preproc1 = StdConv(in_ch, cell_ch, 1, 1, 0)
        self.concat = list(genotype.normal_concat)
        self.dag = nn.ModuleList()
        for edges in genotype.normal:
            row = nn.ModuleList()
            for op_name, state_idx in edges:
                op = build_op(op_name, cell_ch, stride=1)
                if not isinstance(op, Identity):
                    op = nn.Sequential(op, DropPath_())
                op.s_idx = state_idx
                row.append(op)
            self.dag.append(row)

        real_out = len(self.concat) * cell_ch
        self.proj = nn.Identity() if real_out == out_ch else nn.Conv2d(real_out, out_ch, 1, bias=False)

    def forward(self, x):
        states = [self.preproc0(x), self.preproc1(x)]
        for edges in self.dag:
            states.append(sum(op(states[op.s_idx]) for op in edges))
        return self.proj(torch.cat([states[i] for i in self.concat], dim=1))

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)


class DartsUNetSeg(nn.Module):
    """U-Net-like SIRST segmentation network whose encoder blocks are DARTS cells."""

    def __init__(
        self,
        genotype: Genotype,
        genotype_str: str,
        in_ch: int = 3,
        out_ch: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128),
    ):
        super().__init__()
        if len(channels) != 4:
            raise ValueError("DartsUNetSeg currently expects four channel stages")
        self.genotype = genotype
        self.genotype_str = genotype_str
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.channels = tuple(channels)

        c0, c1, c2, c3 = self.channels
        self.stem = ConvBNReLU(in_ch, c0)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.e1 = DartsCell(genotype, c0, c0)
        self.e2 = DartsCell(genotype, c0, c1)
        self.e3 = DartsCell(genotype, c1, c2)
        self.e4 = DartsCell(genotype, c2, c3)

        self.d3 = ConvBNReLU(c3 + c2, c2)
        self.d2 = ConvBNReLU(c2 + c1, c1)
        self.d1 = ConvBNReLU(c1 + c0, c0)
        self.head = nn.Conv2d(c0, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        y3 = self.d3(torch.cat([x3, self.up(x4)], dim=1))
        y2 = self.d2(torch.cat([x2, self.up(y3)], dim=1))
        y1 = self.d1(torch.cat([x1, self.up(y2)], dim=1))
        return self.head(y1)

    def weight_parameters(self):
        return self.parameters()

    def init_model(self, model_init: str = "he_fout", init_div_groups: bool = False):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)

    def save_config(self):
        return {
            "name": self.__class__.__name__,
            "source": "quark0/darts genotype adapted to SIRST segmentation",
            "genotype": self.genotype_str,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "channels": list(self.channels),
        }


def build_darts_sirst_from_json(
    darts_json: str | Path,
    in_ch: int = 3,
    out_ch: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128),
) -> DartsUNetSeg:
    payload = json.loads(Path(darts_json).read_text(encoding="utf-8"))
    genotype_str = payload.get("best_genotype") or payload.get("last_epoch_genotype")
    if not genotype_str:
        raise ValueError(f"genotype not found in {darts_json}")
    genotype = genotype_from_str(genotype_str)
    return DartsUNetSeg(genotype, genotype_str, in_ch=in_ch, out_ch=out_ch, channels=channels)


class OriginalDartsAugmentCell(nn.Module):
    """Original DARTS augmentation cell: two previous states plus normal/reduction genes."""

    def __init__(self, genotype: Genotype, c_pp: int, c_p: int, c: int, reduction_prev: bool, reduction: bool):
        super().__init__()
        self.reduction = reduction
        if reduction_prev:
            self.preproc0 = FactorizedReduce(c_pp, c)
        else:
            self.preproc0 = StdConv(c_pp, c, 1, 1, 0)
        self.preproc1 = StdConv(c_p, c, 1, 1, 0)

        gene = genotype.reduce if reduction else genotype.normal
        self.concat = list(genotype.reduce_concat if reduction else genotype.normal_concat)
        self.dag = nn.ModuleList()
        for edges in gene:
            row = nn.ModuleList()
            for op_name, state_idx in edges:
                stride = 2 if reduction and state_idx < 2 else 1
                op = build_op(op_name, c, stride=stride)
                if not isinstance(op, Identity):
                    op = nn.Sequential(op, DropPath_())
                op.s_idx = state_idx
                row.append(op)
            self.dag.append(row)

    def forward(self, s0, s1):
        states = [self.preproc0(s0), self.preproc1(s1)]
        for edges in self.dag:
            states.append(sum(op(states[op.s_idx]) for op in edges))
        return torch.cat([states[i] for i in self.concat], dim=1)

    def get_flops(self, s0, s1):
        return _count_executed_module_flops(self, s0, s1)


class OriginalDartsDenseSeg(nn.Module):
    """Original DARTS augment network with only the classifier head changed for dense masks."""

    def __init__(
        self,
        genotype: Genotype,
        genotype_str: str,
        in_ch: int = 3,
        out_ch: int = 1,
        init_channels: int = 16,
        layers: int = 8,
        stem_multiplier: int = 3,
    ):
        super().__init__()
        self.genotype = genotype
        self.genotype_str = genotype_str
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.init_channels = init_channels
        self.layers = layers
        self.stem_multiplier = stem_multiplier

        c_cur = stem_multiplier * init_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c_cur, 3, 1, 1, bias=False),
            nn.BatchNorm2d(c_cur),
        )

        c_pp, c_p, c_cur = c_cur, c_cur, init_channels
        reduction_prev = False
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                c_cur *= 2
                reduction = True
            else:
                reduction = False
            cell = OriginalDartsAugmentCell(genotype, c_pp, c_p, c_cur, reduction_prev, reduction)
            reduction_prev = reduction
            self.cells.append(cell)
            c_cur_out = c_cur * len(cell.concat)
            c_pp, c_p = c_p, c_cur_out

        self.head = nn.Conv2d(c_p, out_ch, kernel_size=1)

    def forward(self, x):
        input_size = x.shape[-2:]
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            s0, s1 = s1, cell(s0, s1)
        out = self.head(s1)
        if out.shape[-2:] != input_size:
            out = torch.nn.functional.interpolate(out, size=input_size, mode="bilinear", align_corners=True)
        return out

    def weight_parameters(self):
        return self.parameters()

    def init_model(self, model_init: str = "he_fout", init_div_groups: bool = False):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)

    def save_config(self):
        return {
            "name": self.__class__.__name__,
            "source": "original quark0/darts augment stack with dense segmentation head",
            "genotype": self.genotype_str,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "init_channels": self.init_channels,
            "layers": self.layers,
            "stem_multiplier": self.stem_multiplier,
        }


def build_original_darts_dense_from_json(
    darts_json: str | Path,
    in_ch: int = 3,
    out_ch: int = 1,
    init_channels: int = 16,
    layers: int = 8,
    stem_multiplier: int = 3,
) -> OriginalDartsDenseSeg:
    payload = json.loads(Path(darts_json).read_text(encoding="utf-8"))
    genotype_str = payload.get("best_genotype") or payload.get("last_epoch_genotype")
    if not genotype_str:
        raise ValueError(f"genotype not found in {darts_json}")
    genotype = genotype_from_str(genotype_str)
    return OriginalDartsDenseSeg(
        genotype,
        genotype_str,
        in_ch=in_ch,
        out_ch=out_ch,
        init_channels=init_channels,
        layers=layers,
        stem_multiplier=stem_multiplier,
    )


def decode_phase1_gene(gene_path: str | Path, iterations: int) -> list[list[list[int]]]:
    encoder = []
    for _ in range(iterations):
        encoder.append([[None] for _ in range(iterations - len(encoder))])
    with Path(gene_path).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            parts = line.strip().split("_")
            if len(parts) < 4 or parts[0] != "enc":
                continue
            iteration, layer = int(parts[1]), int(parts[2])
            if iteration == 0:
                pre = [0]
            else:
                pre = [int(v) for v in parts[3:]]
            if iteration < iterations and layer < len(encoder[iteration]):
                encoder[iteration][layer] = pre

    for i, row in enumerate(encoder):
        for j, value in enumerate(row):
            if value is None or value == [None]:
                encoder[i][j] = [0]
    return encoder


def compute_skip_codes(codes: list[list[list[int]]], add_encoder0: bool) -> list[list[bool]]:
    iters = len(codes)
    pre_iter = [0]
    new_codes = []
    for i in range(iters - 1, 0, -1):
        temp = {}
        for level in pre_iter:
            for out in codes[i][level]:
                if out == 0:
                    if level != 0:
                        pre_iter.append(level - 1)
                else:
                    temp[out - 1] = 1
        pre_iter = list(set(temp.keys()))
        new_codes.append(pre_iter)
    new_codes = ([[0]] + new_codes)[::-1]

    skip_codes = [[True for _ in range(iters - m)] for m in range(iters)]
    for i in range(iters):
        if add_encoder0 and i == 0:
            for code in range(iters):
                skip_codes[i][code] = False
            continue
        new_codes[i] = list(set(new_codes[i]))
        for code in new_codes[i]:
            skip_codes[i][code] = False
    return skip_codes


class DartsSameSkeletonBlock(nn.Module):
    """DARTS cell block dropped into the searched segmentation skeleton."""

    def __init__(
        self,
        genotype: Genotype,
        in_ch: int,
        out_ch: int,
        skip_pre: list[int],
        skipped: bool = False,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.skip_pre = sorted(set(skip_pre))
        self.skipped = skipped
        self.operator = None if skipped else DartsCell(genotype, in_ch, out_ch)

    def forward(self, x, skip, encoder_0):
        if self.skipped:
            return x
        _, _, h, w = x.size()
        aligned_skip = [x] + skip
        use = [F.interpolate(aligned_skip[i], size=(h, w), mode="bilinear", align_corners=True) for i in self.skip_pre]
        if encoder_0 != []:
            use.append(F.interpolate(encoder_0, size=(h, w), mode="bilinear", align_corners=True))
        return self.operator(torch.cat(use, dim=1))

    def get_flops(self, x, skip, encoder_0):
        if self.skipped:
            return 0.0, 0.0, x
        return _count_executed_module_flops(self, x, skip, encoder_0)


class DartsSameSkeletonSeg(nn.Module):
    """DARTS cells on the same 5-iteration segmentation skeleton used by the Proxyless runs."""

    def __init__(
        self,
        genotype: Genotype,
        genotype_str: str,
        skeleton_config: str | Path,
        phase1_gene: str | Path,
    ):
        super().__init__()
        config = json.loads(Path(skeleton_config).read_text(encoding="utf-8"))
        self.genotype = genotype
        self.genotype_str = genotype_str
        self.skeleton_config = str(skeleton_config)
        self.phase1_gene = str(phase1_gene)
        self.iterations = int(config["iterations"])
        self.add_encoder0 = str(config.get("add_encoder0", "True")) == "True"
        self.model_name = config.get("model_name", "Super_all")
        self.codes = decode_phase1_gene(phase1_gene, self.iterations)
        self.skip_codes = compute_skip_codes(self.codes, self.add_encoder0)

        blocks = []
        idx = 0
        for iteration in range(self.iterations):
            for layer in range(self.iterations - iteration):
                block_cfg = config["blocks"][idx]
                blocks.append(
                    DartsSameSkeletonBlock(
                        genotype,
                        in_ch=int(block_cfg["in_channels"]),
                        out_ch=int(block_cfg["out_channels"]),
                        skip_pre=self.codes[iteration][layer],
                        skipped=bool(self.skip_codes[iteration][layer]),
                    )
                )
                idx += 1
        self.blocks = nn.ModuleList(blocks)
        post_cfg = config["post_transform_conv_block"]
        self.head = nn.Conv2d(int(post_cfg["in_channels"]), int(post_cfg["out_channels"]), kernel_size=1, bias=True)

    def forward(self, x):
        block_idx = 0
        this_layer = self.iterations
        enc_after = []
        encoder_0 = None
        for iteration in range(self.iterations):
            enc = [None for _ in range(this_layer)]
            if iteration == 0:
                encoder_0 = [None for _ in range(this_layer)]
            for layer in range(this_layer):
                if layer == 0 and iteration == 0:
                    x_in = x
                block = self.blocks[block_idx]
                x_in = block(
                    x_in,
                    enc_after if iteration != 0 else [],
                    encoder_0[layer] if (iteration != 0 and not block.skipped) else [],
                )
                if iteration == 0:
                    encoder_0[layer] = x_in
                elif not block.skipped:
                    encoder_0[layer] = []
                enc[layer] = x_in
                x_in = F.max_pool2d(x_in, 2)
                block_idx += 1
            this_layer -= 1
            enc_after = enc
            x_in = enc_after[0]
        return self.head(x_in)

    def weight_parameters(self):
        return self.parameters()

    def init_model(self, model_init: str = "he_fout", init_div_groups: bool = False):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def get_flops(self, x):
        return _count_executed_module_flops(self, x)

    def save_config(self):
        return {
            "name": self.__class__.__name__,
            "source": "DARTS normal cells on the same searched segmentation skeleton",
            "genotype": self.genotype_str,
            "skeleton_config": self.skeleton_config,
            "phase1_gene": self.phase1_gene,
            "iterations": self.iterations,
            "add_encoder0": str(self.add_encoder0),
            "model_name": self.model_name,
            "blocks": [
                {
                    "in_channels": block.in_ch,
                    "out_channels": block.out_ch,
                    "skip_pre": block.skip_pre,
                    "skipped": block.skipped,
                }
                for block in self.blocks
            ],
        }


def build_darts_same_skeleton_from_json(
    darts_json: str | Path,
    skeleton_config: str | Path,
    phase1_gene: str | Path,
) -> DartsSameSkeletonSeg:
    payload = json.loads(Path(darts_json).read_text(encoding="utf-8"))
    genotype_str = payload.get("best_genotype") or payload.get("last_epoch_genotype")
    if not genotype_str:
        raise ValueError(f"genotype not found in {darts_json}")
    genotype = genotype_from_str(genotype_str)
    return DartsSameSkeletonSeg(genotype, genotype_str, skeleton_config, phase1_gene)
