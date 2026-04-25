"""
LWEU-Net V2 block-level ablation models.

Three variants for isolating EnhancedBlock component contributions:

    LWEUNetV2ExpOnly     : Row A — expansion only
    LWEUNetV2ExpMLP      : Row B — expansion + channel MLP
    LWEUNetV2ExpContext  : Row C — expansion + context gate

All three share identical encoder/bottleneck/decoder structure as V2.
Only the convolutional block type changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lweunet.encoder import (
    StandardResidualBlock, DepthwiseSepConv
)
from src.models.lweunet.bottleneck_v2 import DilatedBottleneckV2
from src.models.lweunet.enhanced_block import GlobalContextGate
from src.models.lweunet.enhanced_block_ablation import (
    EnhancedBlockExpOnly,
    EnhancedBlockExpMLP,
    EnhancedBlockExpContext,
)


# ============================================================
# Shared encoder level (same as encoder_v2.py)
# ============================================================

class EncoderLevelAblation(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.block(x)
        return skip, self.pool(skip)


# ============================================================
# Shared decoder level (same as decoder_v2.py)
# ============================================================

class DecoderLevelAblation(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, block, compress_to=None):
        super().__init__()
        self.compress_to = compress_to
        if compress_to is not None:
            self.compress = nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, compress_to, kernel_size=1, bias=False),
                nn.BatchNorm2d(compress_to),
                nn.SiLU(inplace=True),
            )
        self.block = block

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        if self.compress_to is not None:
            x = self.compress(x)
        return self.block(x)


# ============================================================
# Generic LWEU-Net V2 Ablation Model
# ============================================================

class LWEUNetV2Ablation(nn.Module):
    """
    Generic LWEU-Net V2 with swappable block type.
    Used for all three ablation variants.

    Parameters
    ----------
    block_class  : class   One of EnhancedBlockExpOnly/ExpMLP/ExpContext
    in_channels  : int
    num_classes  : int
    base_filters : int
    dropout_p    : float
    """

    def __init__(
        self,
        block_class,
        in_channels:  int   = 1,
        num_classes:  int   = 4,
        base_filters: int   = 32,
        dropout_p:    float = 0.5,
    ):
        super().__init__()
        f = base_filters

        # Encoder
        self.enc1 = EncoderLevelAblation(StandardResidualBlock(in_channels, f))
        self.enc2 = EncoderLevelAblation(StandardResidualBlock(f, f * 2))
        self.enc3 = EncoderLevelAblation(
            block_class(f * 2, f * 4, expansion=3, dilation=1, use_context=False)
        )
        self.enc4 = EncoderLevelAblation(
            block_class(f * 4, f * 8, expansion=4, dilation=1, use_context=True)
        )

        # Bottleneck — reuse DilatedBottleneckV2 but with ablation block
        # We build it manually to swap block_class
        from src.models.lweunet.enhanced_block import EnhancedBlock
        self.bottleneck = nn.Sequential(
            block_class(f * 8, f * 16, expansion=4, dilation=2, use_context=True),
            nn.Dropout2d(p=dropout_p),
        )

        # Decoder
        self.dec4 = DecoderLevelAblation(
            f * 16, f * 8, f * 8,
            block_class(f * 8, f * 8, expansion=4, dilation=1, use_context=True),
            compress_to=f * 8,
        )
        self.dec3 = DecoderLevelAblation(
            f * 8, f * 4, f * 4,
            block_class(f * 4, f * 4, expansion=3, dilation=1, use_context=True),
            compress_to=f * 4,
        )
        self.dec2 = DecoderLevelAblation(
            f * 4, f * 2, f * 2,
            block_class(f * 4 + f * 2, f * 2, expansion=2, dilation=1, use_context=False),
            compress_to=None,
        )
        self.dec1 = DecoderLevelAblation(
            f * 2, f, f,
            nn.Sequential(DepthwiseSepConv(f * 2 + f, f)),
            compress_to=None,
        )

        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        x        = self.bottleneck(x)
        x        = self.dec4(x, skip4)
        x        = self.dec3(x, skip3)
        x        = self.dec2(x, skip2)
        x        = self.dec1(x, skip1)
        return self.output_conv(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self):
        return {"total": self.count_parameters()}


# ============================================================
# Named model classes for train.py dispatch
# ============================================================

class LWEUNetV2ExpOnly(LWEUNetV2Ablation):
    """Row A — Expansion only (MobileNetV2 contribution)"""
    def __init__(self, **kwargs):
        super().__init__(EnhancedBlockExpOnly, **kwargs)


class LWEUNetV2ExpMLP(LWEUNetV2Ablation):
    """Row B — Expansion + Channel MLP (MobileNetV2 + ConvNeXt)"""
    def __init__(self, **kwargs):
        super().__init__(EnhancedBlockExpMLP, **kwargs)


class LWEUNetV2ExpContext(LWEUNetV2Ablation):
    """Row C — Expansion + Context Gate (MobileNetV2 + SE-Net)"""
    def __init__(self, **kwargs):
        super().__init__(EnhancedBlockExpContext, **kwargs)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    models = [
        (LWEUNetV2ExpOnly,    "Row A — ExpOnly"),
        (LWEUNetV2ExpMLP,     "Row B — ExpMLP"),
        (LWEUNetV2ExpContext, "Row C — ExpContext"),
    ]

    x = torch.randn(2, 1, 224, 224)

    print("LWEU-Net V2 Block Ablation Models")
    print("=" * 50)
    for ModelClass, label in models:
        model  = ModelClass(in_channels=1, num_classes=4, base_filters=32)
        logits = model(x)
        params = model.count_parameters()
        print(f"  [{label}]")
        print(f"    Output : {logits.shape}")
        print(f"    Params : {params:,} ({params/1e6:.2f}M)")
        assert logits.shape == (2, 4, 224, 224)
    print("\n  All passed.")