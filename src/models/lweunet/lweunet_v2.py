"""
LWEU-Net V2 — Enhanced Lightweight U-Net with Novel MBConv-MLP-SE Block

Full model assembly:
    LRHEEncoderV2 → DilatedBottleneckV2 → LWEUNetDecoderV2

Key improvements over V1:
    1. EnhancedBlock replaces DepthwiseSepResidualBlock in deep encoder/decoder
       - Non-linear channel MLP addresses weak cross-channel interaction
       - Expansion ratio creates richer feature space before spatial filtering
       - SiLU/GELU activations improve gradient flow vs ReLU
    2. Global Context Gate at Enc4, Bottleneck, Dec4, Dec3
       - SE-style global average pooling provides whole-slice shape context
       - Directly addresses RV spatial variability problem from ablation
    3. Post-concat compression in Dec4 and Dec3
       - Prevents expensive expansion of 768/384 channel concatenations
       - Stabilises training at deep decoder levels
    4. No standalone attention module (context gate integrated into block)

Parameter budget: ~5-6M (vs 31M baseline — ~5-6× reduction)
FLOPs budget    : ~6-8G (vs 41.86G baseline — ~5-7× reduction)
"""

import torch
import torch.nn as nn

from src.models.lweunet.encoder_v2    import LRHEEncoderV2
from src.models.lweunet.bottleneck_v2 import DilatedBottleneckV2
from src.models.lweunet.decoder_v2    import LWEUNetDecoderV2


class LWEUNetV2(nn.Module):
    """
    LWEU-Net V2.

    Parameters
    ----------
    in_channels  : int    Default 1
    num_classes  : int    Default 4
    base_filters : int    Default 32
    dropout_p    : float  Default 0.5
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        num_classes:  int   = 4,
        base_filters: int   = 32,
        dropout_p:    float = 0.5,
    ):
        super().__init__()

        f = base_filters

        self.encoder = LRHEEncoderV2(
            in_channels  = in_channels,
            base_filters = f,
        )

        self.bottleneck = DilatedBottleneckV2(
            in_channels  = f * 8,    # 256
            out_channels = f * 16,   # 512
            dilation     = 2,
            expansion    = 4,
            dropout_p    = dropout_p,
        )

        self.decoder = LWEUNetDecoderV2(
            base_filters = f,
            num_classes  = num_classes,
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, x = self.encoder(x)
        x        = self.bottleneck(x)
        logits   = self.decoder(skips, x)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        return {
            "encoder":    sum(p.numel() for p in self.encoder.parameters()
                             if p.requires_grad),
            "bottleneck": sum(p.numel() for p in self.bottleneck.parameters()
                             if p.requires_grad),
            "decoder":    sum(p.numel() for p in self.decoder.parameters()
                             if p.requires_grad),
            "total":      self.count_parameters(),
        }


if __name__ == "__main__":
    model  = LWEUNetV2(in_channels=1, num_classes=4, base_filters=32)
    x      = torch.randn(2, 1, 224, 224)
    logits = model(x)
    bd     = model.parameter_breakdown()

    print("LWEU-Net V2 — Full model check")
    print(f"  Input      : {x.shape}")
    print(f"  Output     : {logits.shape}")
    print(f"  Encoder    : {bd['encoder']:>10,} params")
    print(f"  Bottleneck : {bd['bottleneck']:>10,} params")
    print(f"  Decoder    : {bd['decoder']:>10,} params")
    print(f"  Total      : {bd['total']:>10,} params ({bd['total']/1e6:.2f}M)")

    assert logits.shape == (2, 4, 224, 224)
    print("  Passed.")