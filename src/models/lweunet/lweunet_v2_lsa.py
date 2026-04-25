"""
LWEU-Net V2 + LSA — Full model with Lightweight Spatial Attention on skips.
"""

import torch
import torch.nn as nn

from src.models.lweunet.encoder_v2     import LRHEEncoderV2
from src.models.lweunet.bottleneck_v2  import DilatedBottleneckV2
from src.models.lweunet.decoder_v2_lsa import LWEUNetDecoderV2LSA


class LWEUNetV2LSA(nn.Module):
    """
    LWEU-Net V2 with Lightweight Spatial Attention on skip connections.

    Same as LWEUNetV2 but decoder uses LSA at Dec4 and Dec3 skip connections.
    LSA adds ~42 parameters (2 LSA modules × 21 params each) — negligible.
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
            in_channels  = f * 8,
            out_channels = f * 16,
            dilation     = 2,
            expansion    = 4,
            dropout_p    = dropout_p,
        )
        self.decoder = LWEUNetDecoderV2LSA(
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

        # Override LSA conv bias to -1.0 for near-identity init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.in_channels == 2 and m.out_channels == 1:
                nn.init.constant_(m.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips, x = self.encoder(x)
        x        = self.bottleneck(x)
        return self.decoder(skips, x)

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
    model  = LWEUNetV2LSA(in_channels=1, num_classes=4, base_filters=32)
    x      = torch.randn(2, 1, 224, 224)
    logits = model(x)
    bd     = model.parameter_breakdown()

    print("LWEU-Net V2 + LSA")
    print(f"  Output     : {logits.shape}")
    print(f"  Encoder    : {bd['encoder']:>10,}")
    print(f"  Bottleneck : {bd['bottleneck']:>10,}")
    print(f"  Decoder    : {bd['decoder']:>10,}")
    print(f"  Total      : {bd['total']:>10,} ({bd['total']/1e6:.2f}M)")
    assert logits.shape == (2, 4, 224, 224)
    print("  Passed.")