"""
LWEU-Net Encoder — Lightweight Residual Hybrid Encoder (LRHE)

Architecture:
    Level 1 : Standard Residual Conv Block   (1  →  32 filters)
    Level 2 : Standard Residual Conv Block   (32 →  64 filters)
    Level 3 : DepthwiseSep Residual Block    (64  → 128 filters)
    Level 4 : DepthwiseSep Residual Block    (128 → 256 filters)

Design decisions:
    - Levels 1-2 use standard 3×3 convolutions to capture precise low-level
      spatial details (edges, contours, thin MYO boundaries) at high resolution.
    - Levels 3-4 use depthwise separable convolutions for efficient high-level
      semantic feature extraction at lower spatial resolutions.
    - Residual connections throughout ensure stable gradient flow and preserve
      feature information across encoding stages.
    - All convolutions use padding=1 — no spatial shrinkage, no cropping needed
      at skip connections.
    - MaxPool 2×2 for downsampling between levels.
"""

import torch
import torch.nn as nn


# ============================================================
# Standard Residual Conv Block (Levels 1 and 2)
# ============================================================

class StandardResidualBlock(nn.Module):
    """
    Two standard 3×3 Conv → BN → ReLU layers with a residual connection.

    Used in shallow encoder levels (1 and 2) where high spatial resolution
    demands full convolutions to capture fine boundary details critical for
    thin myocardial wall segmentation.

    If in_channels != out_channels, a 1×1 projection conv aligns the
    residual path to match output channels.

    Parameters
    ----------
    in_channels  : int  Input channel count
    out_channels : int  Output channel count
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            # First conv
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second conv
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Residual projection — aligns channels if in != out
        self.residual_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out      = self.conv_block(x)
        return self.relu(out + residual)


# ============================================================
# Depthwise Separable Conv Block (building block for Levels 3 and 4)
# ============================================================

class DepthwiseSepConv(nn.Module):
    """
    Single depthwise separable convolution block.

    Factorises a standard 3×3 Conv into:
        1. Depthwise 3×3 Conv  (one filter per input channel)
        2. Pointwise 1×1 Conv  (linear combination across channels)

    Parameter reduction vs standard conv:
        Standard  : in_ch × out_ch × k × k
        Depthwise : in_ch × k × k  +  in_ch × out_ch
        Reduction : approximately 1/out_ch + 1/k²  ≈ 8-9× for 3×3

    BN → ReLU applied after each sub-convolution.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    dilation     : int  Dilation rate for depthwise conv. Default 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ):
        super().__init__()

        padding = dilation   # keeps spatial size same for any dilation rate

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels,
                kernel_size=3, padding=padding, dilation=dilation,
                groups=in_channels, bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ============================================================
# Depthwise Separable Residual Block (Levels 3 and 4)
# ============================================================

class DepthwiseSepResidualBlock(nn.Module):
    """
    Two DepthwiseSepConv layers with a residual connection.

    Used in deep encoder levels (3 and 4) where feature maps are at low
    spatial resolution and high channel dimensionality. Depthwise separable
    convolutions efficiently learn high-level semantic features at 8-9×
    lower parameter cost than standard convolutions.

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    dilation     : int  Dilation rate for depthwise convs. Default 1.
                        Set to 2 in the bottleneck for expanded receptive field.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            DepthwiseSepConv(in_channels,  out_channels, dilation=dilation),
            DepthwiseSepConv(out_channels, out_channels, dilation=dilation),
        )

        # Residual projection for channel alignment
        self.residual_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels else nn.Identity()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out      = self.conv_block(x)
        return self.relu(out + residual)


# ============================================================
# Encoder Level — conv block + MaxPool (returns skip + downsampled)
# ============================================================

class EncoderLevel(nn.Module):
    """
    Single encoder level: residual block → skip connection + MaxPool.

    Returns both the pre-pool feature map (skip connection for decoder)
    and the post-pool feature map (input to next encoder level).

    Parameters
    ----------
    in_channels  : int
    out_channels : int
    block_type   : str  'standard' for Levels 1-2, 'depthwise' for Levels 3-4
    dilation     : int  Only used when block_type='depthwise'. Default 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_type: str = "standard",
        dilation: int = 1,
    ):
        super().__init__()

        assert block_type in ("standard", "depthwise"), \
            f"block_type must be 'standard' or 'depthwise', got '{block_type}'"

        if block_type == "standard":
            self.block = StandardResidualBlock(in_channels, out_channels)
        else:
            self.block = DepthwiseSepResidualBlock(
                in_channels, out_channels, dilation=dilation
            )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        skip : (B, out_channels, H,   W  )  — for skip connection to decoder
        down : (B, out_channels, H/2, W/2)  — input to next encoder level
        """
        skip = self.block(x)
        down = self.pool(skip)
        return skip, down


# ============================================================
# Full LRHE Encoder
# ============================================================

class LRHEEncoder(nn.Module):
    """
    Lightweight Residual Hybrid Encoder (LRHE).

    Four hierarchical levels:
        Level 1 : StandardResidualBlock  (in_ch  →  32)  224×224
        Level 2 : StandardResidualBlock  (32     →  64)  112×112
        Level 3 : DepthwiseSepResidual   (64     → 128)   56×56
        Level 4 : DepthwiseSepResidual   (128    → 256)   28×28
        Output  :                                          14×14 (fed to bottleneck)

    Parameters
    ----------
    in_channels  : int   Input image channels. Default 1 (grayscale MRI).
    base_filters : int   Filters at Level 1. Default 32.
                         Progression: base × [1, 2, 4, 8]

    Returns (from forward)
    ----------------------
    skips : list of 4 tensors [skip1, skip2, skip3, skip4]
            used by the decoder at matching levels
    out   : tensor — output of Level 4 pooling, fed to bottleneck
    """

    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()

        f = base_filters   # 32

        self.level1 = EncoderLevel(in_channels, f,      block_type="standard")
        self.level2 = EncoderLevel(f,           f * 2,  block_type="standard")
        self.level3 = EncoderLevel(f * 2,       f * 4,  block_type="depthwise")
        self.level4 = EncoderLevel(f * 4,       f * 8,  block_type="depthwise")

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 1, 224, 224)

        Returns
        -------
        skips : [
            (B, 32,  224, 224),   skip1
            (B, 64,  112, 112),   skip2
            (B, 128,  56,  56),   skip3
            (B, 256,  28,  28),   skip4
        ]
        out   : (B, 256, 14, 14)  → fed to bottleneck
        """
        skip1, x = self.level1(x)
        skip2, x = self.level2(x)
        skip3, x = self.level3(x)
        skip4, x = self.level4(x)

        return [skip1, skip2, skip3, skip4], x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    encoder = LRHEEncoder(in_channels=1, base_filters=32)
    x       = torch.randn(2, 1, 224, 224)

    skips, out = encoder(x)

    print("LRHE Encoder — shape check")
    print(f"  Input      : {x.shape}")
    for i, s in enumerate(skips):
        print(f"  Skip {i+1}     : {s.shape}")
    print(f"  Output     : {out.shape}")
    print(f"  Parameters : {encoder.count_parameters():,}")

    # Expected:
    # Skip 1 : (2, 32,  224, 224)
    # Skip 2 : (2, 64,  112, 112)
    # Skip 3 : (2, 128,  56,  56)
    # Skip 4 : (2, 256,  28,  28)
    # Output : (2, 256,  14,  14)