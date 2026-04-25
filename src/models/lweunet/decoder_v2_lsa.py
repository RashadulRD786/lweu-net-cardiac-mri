"""
LWEU-Net V2 Decoder with Lightweight Spatial Attention (LSA)

Changes from decoder_v2.py:
    - LSA applied to skip connections at Dec4 and Dec3
    - Skip features are spatially refined before concatenation
    - Dec2 and Dec1 unchanged (higher resolution, cost not justified)

Flow at Dec4:
    Upsample → LSA(skip4) → Concat → 1×1 compress → EnhancedBlock
Flow at Dec3:
    Upsample → LSA(skip3) → Concat → 1×1 compress → EnhancedBlock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lweunet.enhanced_block import EnhancedBlock
from src.models.lweunet.encoder import DepthwiseSepConv
from src.models.lweunet.lsa import LightweightSpatialAttention


class DecoderLevelV2LSA(nn.Module):
    """
    Decoder level with optional LSA on skip connection.

    Parameters
    ----------
    in_channels   : int
    skip_channels : int
    out_channels  : int
    block         : nn.Module
    compress_to   : int or None
    use_lsa       : bool  Apply LSA to skip before concat. Default False.
    """

    def __init__(
        self,
        in_channels:   int,
        skip_channels: int,
        out_channels:  int,
        block:         nn.Module,
        compress_to:   int  = None,
        use_lsa:       bool = False,
    ):
        super().__init__()

        concat_ch      = in_channels + skip_channels
        self.compress_to = compress_to
        self.use_lsa   = use_lsa

        if use_lsa:
            self.lsa = LightweightSpatialAttention(kernel_size=3)

        if compress_to is not None:
            self.compress = nn.Sequential(
                nn.Conv2d(concat_ch, compress_to, kernel_size=1, bias=False),
                nn.BatchNorm2d(compress_to),
                nn.SiLU(inplace=True),
            )

        self.block = block

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Bilinear upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Optional LSA on skip features
        if self.use_lsa:
            skip = self.lsa(skip)

        # Concatenate
        x = torch.cat([x, skip], dim=1)

        # Optional compression
        if self.compress_to is not None:
            x = self.compress(x)

        return self.block(x)


class LWEUNetDecoderV2LSA(nn.Module):
    """
    LWEU-Net V2 Decoder with LSA at Dec4 and Dec3.

    Parameters
    ----------
    base_filters : int  Default 32
    num_classes  : int  Default 4
    """

    def __init__(self, base_filters: int = 32, num_classes: int = 4):
        super().__init__()

        f = base_filters

        # Dec4: LSA on skip4 → compress 768→256 → EnhancedBlock(r=4,+context)
        self.level4 = DecoderLevelV2LSA(
            in_channels   = f * 16,
            skip_channels = f * 8,
            out_channels  = f * 8,
            compress_to   = f * 8,
            use_lsa       = True,
            block = EnhancedBlock(
                in_channels  = f * 8,
                out_channels = f * 8,
                expansion    = 4,
                use_context  = True,
            ),
        )

        # Dec3: LSA on skip3 → compress 384→128 → EnhancedBlock(r=3,+context)
        self.level3 = DecoderLevelV2LSA(
            in_channels   = f * 8,
            skip_channels = f * 4,
            out_channels  = f * 4,
            compress_to   = f * 4,
            use_lsa       = True,
            block = EnhancedBlock(
                in_channels  = f * 4,
                out_channels = f * 4,
                expansion    = 3,
                use_context  = True,
            ),
        )

        # Dec2: no LSA → EnhancedBlock(r=2)
        self.level2 = DecoderLevelV2LSA(
            in_channels   = f * 4,
            skip_channels = f * 2,
            out_channels  = f * 2,
            compress_to   = None,
            use_lsa       = False,
            block = EnhancedBlock(
                in_channels  = f * 4 + f * 2,
                out_channels = f * 2,
                expansion    = 2,
                use_context  = False,
            ),
        )

        # Dec1: no LSA → DepthwiseSepConv
        self.level1 = DecoderLevelV2LSA(
            in_channels   = f * 2,
            skip_channels = f,
            out_channels  = f,
            compress_to   = None,
            use_lsa       = False,
            block = nn.Sequential(
                DepthwiseSepConv(f * 2 + f, f),
            ),
        )

        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, skips: list, x: torch.Tensor) -> torch.Tensor:
        skip1, skip2, skip3, skip4 = skips
        x = self.level4(x, skip4)
        x = self.level3(x, skip3)
        x = self.level2(x, skip2)
        x = self.level1(x, skip1)
        return self.output_conv(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from src.models.lweunet.encoder_v2 import LRHEEncoderV2

    encoder = LRHEEncoderV2(in_channels=1, base_filters=32)
    decoder = LWEUNetDecoderV2LSA(base_filters=32, num_classes=4)

    x = torch.randn(2, 1, 224, 224)
    skips, enc_out = encoder(x)
    bn_out = torch.randn(2, 512, 14, 14)
    logits = decoder(skips, bn_out)

    print(f"Decoder V2+LSA output : {logits.shape}")
    print(f"Decoder V2+LSA params : {decoder.count_parameters():,}")
    assert logits.shape == (2, 4, 224, 224)
    print("Passed.")