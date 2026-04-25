"""
Block-level ablation variants of EnhancedBlock.

Three variants isolating individual component contributions:

    EnhancedBlockExpOnly     : Expansion + DW + single projection
                               (MobileNetV2 contribution alone)

    EnhancedBlockExpMLP      : Expansion + DW + Channel MLP, no context gate
                               (MobileNetV2 + ConvNeXt, no SE)

    EnhancedBlockExpContext  : Expansion + DW + Context Gate, no channel MLP
                               (MobileNetV2 + SE, no ConvNeXt)

Full EnhancedBlock (all three) is in enhanced_block.py — already implemented.

Ablation comparison:
    Base           → 0.8664  (standard depthwise sep)
    ExpOnly   (A)  → ?       (expansion alone)
    ExpMLP    (B)  → ?       (expansion + channel MLP)
    ExpContext (C) → ?       (expansion + context gate)
    V2 Full        → 0.8937  (all three combined)
"""

import torch
import torch.nn as nn
from src.models.lweunet.enhanced_block import GlobalContextGate


# ============================================================
# Row A — Expansion Only (MobileNetV2 contribution)
# ============================================================

class EnhancedBlockExpOnly(nn.Module):
    """
    Expansion + Depthwise + Single linear projection.
    No channel MLP, no context gate.
    Isolates the MobileNetV2 inverted residual contribution.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        expansion:    int  = 4,
        dilation:     int  = 1,
        use_context:  bool = False,   # ignored — kept for API compatibility
    ):
        super().__init__()

        mid = in_channels * expansion
        pad = dilation

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=pad,
                      dilation=dilation, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        # Single linear projection — no MLP, no non-linearity
        self.project = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

        self.post_bn = nn.BatchNorm2d(out_channels)
        self.residual_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out      = self.expand(x)
        out      = self.dw(out)
        out      = self.project(out)
        return self.post_bn(out + residual)


# ============================================================
# Row B — Expansion + Channel MLP (MobileNetV2 + ConvNeXt)
# ============================================================

class EnhancedBlockExpMLP(nn.Module):
    """
    Expansion + Depthwise + Channel MLP. No context gate.
    Isolates MobileNetV2 + ConvNeXt combined contribution.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        expansion:    int  = 4,
        dilation:     int  = 1,
        use_context:  bool = False,   # ignored — kept for API compatibility
    ):
        super().__init__()

        mid = in_channels * expansion
        pad = dilation

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=pad,
                      dilation=dilation, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
        )
        # Channel MLP — non-linear cross-channel interaction
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
        )
        # No context gate

        self.post_bn = nn.BatchNorm2d(out_channels)
        self.residual_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out      = self.expand(x)
        out      = self.dw(out)
        out      = self.channel_mlp(out)
        return self.post_bn(out + residual)


# ============================================================
# Row C — Expansion + Context Gate (MobileNetV2 + SE-Net)
# ============================================================

class EnhancedBlockExpContext(nn.Module):
    """
    Expansion + Depthwise + Context Gate. No channel MLP.
    Isolates MobileNetV2 + SE-Net combined contribution.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        expansion:    int  = 4,
        dilation:     int  = 1,
        use_context:  bool = False,
    ):
        super().__init__()

        mid = in_channels * expansion
        pad = dilation

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=3, padding=pad,
                      dilation=dilation, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
        )
        # Single projection (no MLP)
        self.project = nn.Conv2d(mid, out_channels, kernel_size=1, bias=False)

        # Context gate always applied when use_context=True
        self.use_context = use_context
        if use_context:
            self.context_gate = GlobalContextGate(out_channels, reduction=4)

        self.post_bn = nn.BatchNorm2d(out_channels)
        self.residual_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out      = self.expand(x)
        out      = self.dw(out)
        out      = self.project(out)
        if self.use_context:
            out = self.context_gate(out)
        return self.post_bn(out + residual)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    print("Block ablation variants — shape check")
    print("-" * 50)

    variants = [
        (EnhancedBlockExpOnly,    "Row A — ExpOnly"),
        (EnhancedBlockExpMLP,     "Row B — ExpMLP"),
        (EnhancedBlockExpContext, "Row C — ExpContext"),
    ]

    for BlockClass, label in variants:
        block  = BlockClass(128, 256, expansion=4, dilation=1, use_context=True)
        x      = torch.randn(2, 128, 28, 28)
        y      = block(x)
        params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        print(f"  [{label}]")
        print(f"    {x.shape} → {y.shape} | params={params:,}")
        assert y.shape == (2, 256, 28, 28)

    print("\n  All passed.")