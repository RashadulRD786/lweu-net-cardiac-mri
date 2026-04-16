"""
LWEU-Net Decoder

Architecture (bottom-up):
    Level 4 : Bilinear ×2 → Concat skip4 → DepthwiseSep → ECA
    Level 3 : Bilinear ×2 → Concat skip3 → DepthwiseSep → ECA
    Level 2 : Bilinear ×2 → Concat skip2 → DepthwiseSep → ECA
    Level 1 : Bilinear ×2 → Concat skip1 → DepthwiseSep (no ECA)
    Output  : Conv1×1 → Softmax (4 classes)

Channel progression:
    Level 4 : 512 + 256 = 768 → 256  (ECA)
    Level 3 : 256 + 128 = 384 → 128  (ECA)
    Level 2 : 128 +  64 = 192 →  64  (ECA)
    Level 1 :  64 +  32 =  96 →  32  (no ECA)
    Output  :  32 → 4

Design decisions:
    - Bilinear upsample: no learnable parameters, no checkerboard artifacts
    - padding=1 on all convolutions: no spatial shrinkage, no cropping needed
    - DepthwiseSep refinement: reduces parameters significantly vs standard conv
    - ECA attention on Levels 4-2: dynamically amplifies boundary-sensitive
      channels and suppresses irrelevant ones — improves MYO boundary precision
    - No ECA at Level 1: full-resolution features contain precise low-level
      spatial details that should not be selectively suppressed before output
    - Softmax output: consistent with CrossEntropyLoss training setup

ECA (Efficient Channel Attention) — Wang et al. 2020:
    - Global average pooling → 1D conv (kernel adaptive to channel count)
    - Sigmoid gate → channel-wise multiplication
    - No dimensionality reduction (unlike SE-Net) → more efficient
    - Kernel size k determined by: k = |log2(C)/2 + 0.5)| rounded to odd
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.lweunet.encoder import DepthwiseSepConv


# ============================================================
# ECA Attention Module
# ============================================================

class ECAAttention(nn.Module):
    """
    Efficient Channel Attention (Wang et al., 2020).

    Recalibrates channel-wise feature responses using a lightweight
    1D convolution on the channel descriptor vector. No dimensionality
    reduction — more parameter-efficient than SE-Net.

    The 1D conv kernel size is computed adaptively from channel count:
        k = nearest odd integer to |log2(C) / 2 + 0.5|

    Parameters
    ----------
    channels : int  Number of input/output channels (unchanged by ECA)
    """

    def __init__(self, channels: int):
        super().__init__()

        # Adaptive kernel size from channel count
        k = int(abs(math.log2(channels) / 2 + 0.5))
        k = k if k % 2 == 1 else k + 1   # ensure odd for symmetric padding

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d   = nn.Conv1d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = k,
            padding      = k // 2,
            bias         = False,
        )
        self.sigmoid  = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        out : (B, C, H, W)  — channel-recalibrated feature map
        """
        # Channel descriptor: (B, C, 1, 1) → (B, 1, C)
        y = self.avg_pool(x)                      # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)       # (B, 1, C)

        # 1D conv across channels
        y = self.conv1d(y)                        # (B, 1, C)

        # Gate: (B, C, 1, 1)
        y = self.sigmoid(y)
        y = y.transpose(-1, -2).unsqueeze(-1)     # (B, C, 1, 1)

        return x * y.expand_as(x)


# ============================================================
# Depthwise Separable Refinement Block (decoder conv block)
# ============================================================

class DecoderConvBlock(nn.Module):
    """
    Two DepthwiseSepConv layers for feature refinement in the decoder.

    Reduces concatenated channels (skip + upsampled) down to target
    output channels efficiently.

    Parameters
    ----------
    in_channels  : int  Channels after skip concatenation
    out_channels : int  Target output channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.block = nn.Sequential(
            DepthwiseSepConv(in_channels,  out_channels),
            DepthwiseSepConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ============================================================
# Single Decoder Level
# ============================================================

class DecoderLevel(nn.Module):
    """
    Single decoder level:
        Bilinear upsample ×2
        → Concatenate skip connection
        → DepthwiseSep refinement
        → ECA attention (optional)

    Parameters
    ----------
    in_channels   : int   Channels from previous decoder level (or bottleneck)
    skip_channels : int   Channels from matching encoder skip connection
    out_channels  : int   Output channels after refinement
    use_eca       : bool  Whether to apply ECA after refinement. Default True.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_eca: bool = True,
    ):
        super().__init__()

        self.use_eca = use_eca

        # Refinement: processes concatenated (in_channels + skip_channels)
        self.conv = DecoderConvBlock(
            in_channels  = in_channels + skip_channels,
            out_channels = out_channels,
        )

        # ECA applied after refinement
        if use_eca:
            self.eca = ECAAttention(channels=out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, in_channels,   H/2, W/2)  from previous level
        skip : (B, skip_channels, H,   W  )  from matching encoder level

        Returns
        -------
        out  : (B, out_channels,  H,   W  )
        """
        # Step 1 — Bilinear upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Step 2 — Concatenate skip (no cropping needed — padding=1 everywhere)
        x = torch.cat([x, skip], dim=1)

        # Step 3 — Depthwise separable refinement
        x = self.conv(x)

        # Step 4 — ECA attention (optional)
        if self.use_eca:
            x = self.eca(x)

        return x


# ============================================================
# Full Decoder
# ============================================================

class LWEUNetDecoder(nn.Module):
    """
    LWEU-Net Decoder.

    Four levels mirroring the encoder, plus a 1×1 output convolution.

    Parameters
    ----------
    base_filters : int   Must match encoder base_filters. Default 32.
    num_classes  : int   Output segmentation classes. Default 4.

    Expected inputs
    ---------------
    skips : [skip1, skip2, skip3, skip4] from LRHEEncoder
    x     : bottleneck output (B, 512, 14, 14)
    """

    def __init__(self, base_filters: int = 32, num_classes: int = 4,use_eca: bool = True):
        super().__init__()

        f = base_filters   # 32

        # Level 4: 512 + 256 → 256, ECA
        self.level4 = DecoderLevel(
            in_channels   = f * 16,   # 512 from bottleneck
            skip_channels = f * 8,    # 256 from encoder skip4
            out_channels  = f * 8,    # 256
            use_eca       = use_eca ,
        )

        # Level 3: 256 + 128 → 128, ECA
        self.level3 = DecoderLevel(
            in_channels   = f * 8,    # 256
            skip_channels = f * 4,    # 128
            out_channels  = f * 4,    # 128
            use_eca       = use_eca ,
        )

        # Level 2: 128 + 64 → 64, ECA
        self.level2 = DecoderLevel(
            in_channels   = f * 4,    # 128
            skip_channels = f * 2,    # 64
            out_channels  = f * 2,    # 64
            use_eca       = use_eca ,
        )

        # Level 1: 64 + 32 → 32, no ECA
        self.level1 = DecoderLevel(
            in_channels   = f * 2,    # 64
            skip_channels = f,        # 32
            out_channels  = f,        # 32
            use_eca       = False,
        )

        # Output: 32 → num_classes
        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, skips: list, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        skips : [skip1, skip2, skip3, skip4]
                skip1: (B, 32,  224, 224)
                skip2: (B, 64,  112, 112)
                skip3: (B, 128,  56,  56)
                skip4: (B, 256,  28,  28)
        x     : (B, 512, 14, 14)  bottleneck output

        Returns
        -------
        logits : (B, num_classes, 224, 224)  raw logits (before softmax)
        """
        skip1, skip2, skip3, skip4 = skips

        x = self.level4(x, skip4)   # (B, 256, 28, 28)
        x = self.level3(x, skip3)   # (B, 128, 56, 56)
        x = self.level2(x, skip2)   # (B,  64,112,112)
        x = self.level1(x, skip1)   # (B,  32,224,224)

        return self.output_conv(x)  # (B, num_classes, 224, 224)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    from src.models.lweunet.encoder import LRHEEncoder

    encoder = LRHEEncoder(in_channels=1, base_filters=32)
    decoder = LWEUNetDecoder(base_filters=32, num_classes=4)

    x              = torch.randn(2, 1, 224, 224)
    skips, enc_out = encoder(x)

    # Simulate bottleneck output (512 channels)
    bottleneck_out = torch.randn(2, 512, 14, 14)

    logits = decoder(skips, bottleneck_out)

    print("LWEU-Net Decoder — shape check")
    print(f"  Encoder output : {enc_out.shape}")
    print(f"  Bottleneck out : {bottleneck_out.shape}")
    print(f"  Decoder output : {logits.shape}")
    print(f"  Decoder params : {decoder.count_parameters():,}")

    assert logits.shape == (2, 4, 224, 224), "Output shape mismatch!"
    print("  Sanity check passed.")