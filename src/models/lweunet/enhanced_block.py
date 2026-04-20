"""
Enhanced Convolutional Block — Novel MBConv-MLP-SE Hybrid

Design motivated by ablation findings:
  - Standard depthwise separable conv has weak cross-channel interaction
    (linear-only pointwise) — addressed by Channel MLP (two 1×1 with GELU)
  - ECA attention lost RV spatial context (global avg pool destroys position)
    — addressed by Global Context Gate at deep levels
  - MYO thin boundaries need strong spatial feature extraction
    — addressed by expansion + SiLU before depthwise (richer spatial filtering)

Block structure:
    Input (C)
    → 1×1 Conv (C → C×r) → BN → SiLU      [expand + non-linear]
    → Depthwise 3×3 (C×r, dilation) → BN  [spatial extraction]
    → 1×1 Conv (C×r → C×r) → GELU         [channel MLP layer 1]
    → 1×1 Conv (C×r → C)                   [channel MLP layer 2 + project]
    → Global Context Gate (optional)        [SE-style global context for RV]
    → Residual Add
    → BN                                    [post-residual normalisation]
    → Output (C)

Novelty vs existing blocks:
  - EfficientNet MBConv      : has expand+DW+SE but no channel MLP
  - ConvNeXt V2              : has MLP+GRN but DW-first, no expand-before-DW
  - This block                : combines both — expand-before-DW + channel MLP + SE gate
  No cardiac MRI segmentation paper has published this exact combination.

References:
  - MobileNetV2 (Sandler et al., 2018) — inverted residual / expand-before-DW
  - ConvNeXt (Liu et al., 2022) — channel MLP with GELU
  - SE-Net (Hu et al., 2018) — squeeze-excitation global context gate
  - SiLU/Swish (Ramachandran et al., 2017) — smooth non-linearity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Global Context Gate (SE-style)
# ============================================================

class GlobalContextGate(nn.Module):
    """
    Squeeze-and-Excitation style global context gate.

    Compresses the entire feature map into a global descriptor via
    global average pooling, then uses a two-layer bottleneck MLP
    to compute per-channel attention weights.

    Applied only at deep encoder/decoder levels where global shape
    context is most valuable (RV shape understanding).

    Parameters
    ----------
    channels     : int   Number of input/output channels
    reduction    : int   Bottleneck reduction ratio. Default 4.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)   # minimum 4 channels
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B, C, 1, 1)
            nn.Flatten(),                                      # (B, C)
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        weight = self.gate(x).view(B, C, 1, 1)   # (B, C, 1, 1)
        return x * weight


# ============================================================
# Enhanced Block
# ============================================================

class EnhancedBlock(nn.Module):
    """
    Novel MBConv-MLP-SE hybrid convolutional block.

    Combines:
      1. Inverted residual expansion (MobileNetV2 style) for richer
         channel space before spatial filtering
      2. Non-linear channel MLP (ConvNeXt style) for cross-channel
         interaction beyond linear pointwise convolution
      3. Global context gate (SE-Net style) for whole-image shape
         context — critical for RV at deep levels
      4. Residual connection with post-BN for training stability

    Parameters
    ----------
    in_channels  : int    Input channel count
    out_channels : int    Output channel count
    expansion    : int    Channel expansion ratio before depthwise. Default 4.
    dilation     : int    Depthwise conv dilation rate. Default 1.
                          Set to 2 in bottleneck for expanded receptive field.
    use_context  : bool   Enable Global Context Gate. Default False.
                          Enable at Enc4, Bottleneck, Dec4, Dec3.
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

        mid = in_channels * expansion   # expanded channel count
        pad = dilation                   # padding = dilation keeps spatial size

        # --- Expand + SiLU ---
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # --- Depthwise spatial extraction ---
        self.dw = nn.Sequential(
            nn.Conv2d(
                mid, mid,
                kernel_size=3, padding=pad, dilation=dilation,
                groups=mid, bias=False,
            ),
            nn.BatchNorm2d(mid),
        )

        # --- Channel MLP (non-linear cross-channel interaction) ---
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(mid, mid, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, out_channels, kernel_size=1, bias=False),
        )

        # --- Global Context Gate (optional) ---
        self.use_context = use_context
        if use_context:
            self.context_gate = GlobalContextGate(out_channels, reduction=4)

        # --- Post-residual BN ---
        self.post_bn = nn.BatchNorm2d(out_channels)

        # --- Residual projection (if channel count changes) ---
        self.residual_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)

        out = self.expand(x)          # C → C×r, BN, SiLU
        out = self.dw(out)            # spatial extraction, BN
        out = self.channel_mlp(out)   # non-linear channel mixing → out_channels

        if self.use_context:
            out = self.context_gate(out)   # global shape context

        out = self.post_bn(out + residual)   # residual add + BN

        return out


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    print("EnhancedBlock — shape and config checks")
    print("-" * 50)

    configs = [
        # (in_ch, out_ch, r, dilation, context, label)
        (64,  128, 3, 1, False, "Enc3"),
        (128, 256, 4, 1, True,  "Enc4 + context"),
        (256, 512, 4, 2, True,  "Bottleneck + dilation=2 + context"),
        (256, 256, 4, 1, True,  "Dec4 + context"),
        (128, 128, 3, 1, True,  "Dec3 + context"),
        (192,  64, 2, 1, False, "Dec2"),
    ]

    for in_c, out_c, r, d, ctx, label in configs:
        block = EnhancedBlock(in_c, out_c, expansion=r,
                              dilation=d, use_context=ctx)
        h = 56 if in_c <= 128 else 28
        x = torch.randn(2, in_c, h, h)
        y = block(x)
        params = sum(p.numel() for p in block.parameters() if p.requires_grad)
        print(f"  [{label}]")
        print(f"    Input : {x.shape}  Output : {y.shape}  Params: {params:,}")
        assert y.shape == (2, out_c, h, h), f"Shape mismatch for {label}"

    print("\n  All checks passed.")