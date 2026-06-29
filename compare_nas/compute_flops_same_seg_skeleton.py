#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile


class DartsCellBlock(nn.Module):
    def __init__(self, genotype, in_ch: int, out_ch: int):
        super().__init__()
        # AugmentCell output channels are 4 * C when concat=range(2, 6).
        cell_c = max(1, out_ch // 4)
        from models.augment_cells import AugmentCell

        self.cell = AugmentCell(
            genotype=genotype,
            C_pp=in_ch,
            C_p=in_ch,
            C=cell_c,
            reduction_p=False,
            reduction=False,
        )
        self.proj = nn.Identity()
        real_out = 4 * cell_c
        if real_out != out_ch:
            self.proj = nn.Conv2d(real_out, out_ch, kernel_size=1, bias=False)

    def forward(self, x):
        y = self.cell(x, x)
        return self.proj(y)


class DartsUNetSeg(nn.Module):
    """Use DARTS cells inside a U-Net-like segmentation skeleton."""

    def __init__(self, genotype, in_ch=3, out_ch=1, ch=(16, 32, 64, 128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.e1 = DartsCellBlock(genotype, ch[0], ch[0])
        self.e2 = DartsCellBlock(genotype, ch[0], ch[1])
        self.e3 = DartsCellBlock(genotype, ch[1], ch[2])
        self.e4 = DartsCellBlock(genotype, ch[2], ch[3])

        self.d3 = nn.Sequential(
            nn.Conv2d(ch[3] + ch[2], ch[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[2]),
            nn.ReLU(inplace=True),
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(ch[2] + ch[1], ch[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[1]),
            nn.ReLU(inplace=True),
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(ch[1] + ch[0], ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(ch[0], out_ch, kernel_size=1)

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


def compute(model, size=256):
    model.eval()
    x = torch.randn(1, 3, size, size)
    flops, params = profile(model, inputs=(x,), verbose=False)
    return flops, params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--darts-json", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=256)
    args = parser.parse_args()

    payload = json.loads(args.darts_json.read_text(encoding="utf-8"))
    geno_str = payload.get("best_genotype") or payload.get("last_epoch_genotype")
    if not geno_str:
        raise SystemExit("genotype not found in darts json")

    sys.path.insert(0, "/home/intern/proxylessnas/third_party/pt.darts")
    from genotypes import from_str

    genotype = from_str(geno_str)
    darts_seg = DartsUNetSeg(genotype)
    darts_flops, darts_params = compute(darts_seg, size=args.input_size)

    spec = importlib.util.spec_from_file_location(
        "proxyless_model_res_unet",
        "/home/intern/proxylessnas/search/models/model_res_Unet.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    res_UNet = module.res_UNet

    proxyless_ref = res_UNet(
        num_classes=1,
        input_channels=3,
        block="Res_block",
        num_blocks=[2, 2, 2, 2],
        nb_filter=[16, 32, 64, 128, 256],
        layer=4,
    )
    proxy_flops, proxy_params = compute(proxyless_ref, size=args.input_size)

    print("Same-skeleton segmentation FLOPs (input: 1x3x{}x{})".format(args.input_size, args.input_size))
    print("DARTS_adapted_FLOPs_M={:.3f}".format(darts_flops / 1e6))
    print("DARTS_adapted_Params_M={:.3f}".format(darts_params / 1e6))
    print("Proxyless_ref_FLOPs_M={:.3f}".format(proxy_flops / 1e6))
    print("Proxyless_ref_Params_M={:.3f}".format(proxy_params / 1e6))
    print("Ratio_DARTS_vs_Proxyless={:.3f}".format(darts_flops / proxy_flops))


if __name__ == "__main__":
    main()
