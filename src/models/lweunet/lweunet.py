"""
LWEU-Net — Lightweight Enhanced U-Net for Cardiac MRI Segmentation

Full model assembly connecting:
    LRHEEncoder     → DilatedBottleneck → LWEUNetDecoder

Architecture summary:
    Input  : (B, 1, 224, 224)  grayscale MRI slice
    Encoder: 4-level LRHE
             L1: StandardResidual  (1   →  32)  224×224
             L2: StandardResidual  (32  →  64)  112×112
             L3: DepthwiseSepRes   (64  → 128)   56×56
             L4: DepthwiseSepRes   (128 → 256)   28×28
    Bottleneck: DilatedDepthwiseSep (256 → 512, dilation=2)  14×14
    Decoder: 4-level with optional ECA
             L4: 512+256 → 256  (ECA optional)  28×28
             L3: 256+128 → 128  (ECA optional)  56×56
             L2: 128+ 64 →  64  (ECA optional) 112×112
             L1:  64+ 32 →  32  (no ECA)       224×224
    Output : Conv1×1 → 4 classes → raw logits

Ablation control via flags:
    use_eca=False  → Row 2 (LWEU-Net Base)
    use_eca=True   → Row 3 (+ Decoder ECA) and Row 4 (Full)
    Boundary loss is controlled separately in combo_loss.py

Parameter budget (approximate):
    Encoder    : ~240K
    Bottleneck : ~536K
    Decoder    : ~371K
    Total      : ~1.15M  (vs 31.04M baseline — ~27× reduction)
"""

import torch
import torch.nn as nn

from src.models.lweunet.encoder    import LRHEEncoder
from src.models.lweunet.bottleneck import DilatedBottleneck
from src.models.lweunet.decoder    import LWEUNetDecoder


class LWEUNet(nn.Module):
    """
    Lightweight Enhanced U-Net (LWEU-Net).

    Parameters
    ----------
    in_channels  : int    Input image channels. Default 1.
    num_classes  : int    Output segmentation classes. Default 4.
    base_filters : int    Base filter count for encoder/decoder. Default 32.
    dropout_p    : float  Bottleneck dropout probability. Default 0.5.
    use_eca      : bool   Enable ECA attention in decoder. Default True.
                          Set False for ablation Row 2 (LWEU-Net Base).
    """

    def __init__(
        self,
        in_channels:  int   = 1,
        num_classes:  int   = 4,
        base_filters: int   = 32,
        dropout_p:    float = 0.5,
        use_eca:      bool  = True,
    ):
        super().__init__()

        self.use_eca = use_eca
        f = base_filters   # 32

        self.encoder = LRHEEncoder(
            in_channels  = in_channels,
            base_filters = f,
        )

        self.bottleneck = DilatedBottleneck(
            in_channels  = f * 8,    # 256 — encoder Level 4 output
            out_channels = f * 16,   # 512
            dilation     = 2,
            dropout_p    = dropout_p,
        )

        self.decoder = LWEUNetDecoder(
            base_filters = f,
            num_classes  = num_classes,
            use_eca      = use_eca,
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self):
        """He normal for Conv2d layers, constant for BatchNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x      : (B, 1, 224, 224)  normalised grayscale MRI slice

        Returns
        -------
        logits : (B, 4, 224, 224)  raw logits — apply softmax for probs
        """
        skips, x = self.encoder(x)
        x        = self.bottleneck(x)
        logits   = self.decoder(skips, x)
        return logits

    def count_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Parameter count per component — useful for thesis reporting."""
        return {
            "encoder":    sum(p.numel() for p in self.encoder.parameters()
                             if p.requires_grad),
            "bottleneck": sum(p.numel() for p in self.bottleneck.parameters()
                             if p.requires_grad),
            "decoder":    sum(p.numel() for p in self.decoder.parameters()
                             if p.requires_grad),
            "total":      self.count_parameters(),
        }


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("LWEU-Net Full Model — Sanity Check")
    print("=" * 55)

    for eca_setting in [False, True]:
        model  = LWEUNet(
            in_channels  = 1,
            num_classes  = 4,
            base_filters = 32,
            dropout_p    = 0.5,
            use_eca      = eca_setting,
        )
        x      = torch.randn(2, 1, 224, 224)
        logits = model(x)

        breakdown = model.parameter_breakdown()
        label     = "LWEU-Net Base" if not eca_setting else "LWEU-Net + ECA"

        print(f"\n  [{label}]  use_eca={eca_setting}")
        print(f"  Input      : {x.shape}")
        print(f"  Output     : {logits.shape}")
        print(f"  Encoder    : {breakdown['encoder']:>10,} params")
        print(f"  Bottleneck : {breakdown['bottleneck']:>10,} params")
        print(f"  Decoder    : {breakdown['decoder']:>10,} params")
        print(f"  Total      : {breakdown['total']:>10,} params  "
              f"({breakdown['total']/1e6:.2f}M)")

        assert logits.shape == (2, 4, 224, 224), "Shape mismatch!"

    print("\n  Sanity check passed for both ablation variants.")
    print("=" * 55)