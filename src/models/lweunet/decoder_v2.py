"""
LWEU-Net V2 Decoder

Architecture (bottom-up):
    Dec4: Upsample → Concat → 1×1 compress(768→256) → EnhancedBlock(r=4, +context)
    Dec3: Upsample → Concat → 1×1 compress(384→128) → EnhancedBlock(r=3, +context)
    Dec2: Upsample → Concat → EnhancedBlock(r=2, in=192, out=64)
    Dec1: Upsample → Concat → DepthwiseSepConv(96→32)
    Out : Conv1×1 → 4 classes

Changes from V1:
    - Dec4/Dec3 use 1×1 compression before EnhancedBlock
      (prevents expensive 768/384ch expansion, stabilises training)
    - Dec4/Dec3 use EnhancedBlock with global context
    - Dec2 uses EnhancedBlock without context (higher resolution, lighter)
    - Dec1 uses standard depthwise separable conv (full resolution, lightweight)
    - No ECA attention (replaced by integrated context gate in EnhancedBlock)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lweunet.enhanced_block import EnhancedBlock
from src.models.lweunet.encoder import DepthwiseSepConv


class DecoderLevelV2(nn.Module):
    """
    Single decoder level for V2.

    For Dec4 and Dec3: includes 1×1 compression before EnhancedBlock.
    For Dec2: EnhancedBlock directly on concatenated features.
    For Dec1: DepthwiseSepConv on concatenated features.

    Parameters
    ----------
    in_channels     : int    Channels from previous decoder level (before upsample)
    skip_channels   : int    Channels from encoder skip connection
    out_channels    : int    Output channels
    block           : nn.Module  Pre-constructed processing block
    compress_to     : int or None
                      If not None, apply 1×1 compression from
                      (in_channels+skip_channels) → compress_to before block.
    """

    def __init__(
        self,
        in_channels:   int,
        skip_channels: int,
        out_channels:  int,
        block:         nn.Module,
        compress_to:   int = None,
    ):
        super().__init__()

        concat_ch = in_channels + skip_channels
        self.compress_to = compress_to

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

        # Concatenate skip (no cropping needed — padding=1 everywhere)
        x = torch.cat([x, skip], dim=1)

        # Optional compression
        if self.compress_to is not None:
            x = self.compress(x)

        return self.block(x)


class LWEUNetDecoderV2(nn.Module):
    """
    LWEU-Net V2 Decoder.

    Parameters
    ----------
    base_filters : int   Must match encoder base_filters. Default 32.
    num_classes  : int   Output classes. Default 4.
    """

    def __init__(self, base_filters: int = 32, num_classes: int = 4):
        super().__init__()

        f = base_filters  # 32

        # Dec4: 512 + 256 = 768 → compress to 256 → EnhancedBlock(r=4, +context)
        self.level4 = DecoderLevelV2(
            in_channels   = f * 16,   # 512 from bottleneck
            skip_channels = f * 8,    # 256 from skip4
            out_channels  = f * 8,    # 256
            compress_to   = f * 8,    # compress 768 → 256
            block = EnhancedBlock(
                in_channels  = f * 8,   # 256
                out_channels = f * 8,   # 256
                expansion    = 4,
                use_context  = True,
            ),
        )

        # Dec3: 256 + 128 = 384 → compress to 128 → EnhancedBlock(r=3, +context)
        self.level3 = DecoderLevelV2(
            in_channels   = f * 8,    # 256
            skip_channels = f * 4,    # 128
            out_channels  = f * 4,    # 128
            compress_to   = f * 4,    # compress 384 → 128
            block = EnhancedBlock(
                in_channels  = f * 4,   # 128
                out_channels = f * 4,   # 128
                expansion    = 3,
                use_context  = True,
            ),
        )

        # Dec2: 128 + 64 = 192 → EnhancedBlock(r=2, no context)
        self.level2 = DecoderLevelV2(
            in_channels   = f * 4,    # 128
            skip_channels = f * 2,    # 64
            out_channels  = f * 2,    # 64
            compress_to   = None,     # no compression
            block = EnhancedBlock(
                in_channels  = f * 4 + f * 2,   # 192 (concat, no compress)
                out_channels = f * 2,            # 64
                expansion    = 2,
                use_context  = False,
            ),
        )

        # Dec1: 64 + 32 = 96 → DepthwiseSepConv(96→32)
        self.level1 = DecoderLevelV2(
            in_channels   = f * 2,   # 64
            skip_channels = f,        # 32
            out_channels  = f,        # 32
            compress_to   = None,
            block = nn.Sequential(
                DepthwiseSepConv(f * 2 + f, f),   # 96 → 32
            ),
        )

        # Output
        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, skips: list, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        skips : [skip1(B,32,224,224), skip2(B,64,112,112),
                 skip3(B,128,56,56),  skip4(B,256,28,28)]
        x     : (B, 512, 14, 14) bottleneck output

        Returns
        -------
        logits : (B, num_classes, 224, 224)
        """
        skip1, skip2, skip3, skip4 = skips

        x = self.level4(x, skip4)   # (B, 256, 28,  28)
        x = self.level3(x, skip3)   # (B, 128, 56,  56)
        x = self.level2(x, skip2)   # (B,  64, 112, 112)
        x = self.level1(x, skip1)   # (B,  32, 224, 224)

        return self.output_conv(x)  # (B, num_classes, 224, 224)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from src.models.lweunet.encoder_v2 import LRHEEncoderV2

    encoder = LRHEEncoderV2(in_channels=1, base_filters=32)
    decoder = LWEUNetDecoderV2(base_filters=32, num_classes=4)

    x = torch.randn(2, 1, 224, 224)
    skips, enc_out = encoder(x)
    bottleneck_out = torch.randn(2, 512, 14, 14)
    logits = decoder(skips, bottleneck_out)

    print(f"Decoder V2 output : {logits.shape}")
    print(f"Decoder V2 params : {decoder.count_parameters():,}")
    assert logits.shape == (2, 4, 224, 224)
    print("Passed.")