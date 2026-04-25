"""
Lightweight Spatial Attention (LSA) for Skip Connections

Applied to encoder skip features before concatenation with decoder features.
Specifically targets RV spatial variability — suppresses irrelevant spatial
regions (absent RV in apical/basal slices) while preserving relevant ones.

Design:
    Input skip: (B, C, H, W)
    AvgPool (channel-wise) → (B, 1, H, W)   average feature presence
    MaxPool (channel-wise) → (B, 1, H, W)   peak feature presence
    Concat                 → (B, 2, H, W)
    Conv(2→1, 3×3, pad=1)  → (B, 1, H, W)
    BN                     → stable attention map
    Sigmoid                → S ∈ [0, 1]
    Output = skip × (1 + S)               residual gating

Design decisions:
    - Dual pooling (avg+max): captures complementary spatial statistics
      avg pooling: where features are consistently present
      max pooling: where at least one channel is strongly activated
    - BN after conv: stabilises attention map, prevents noisy masks early
    - Residual gating (1+S): preserves weak features, never suppresses
      below 1× — prevents the ECA problem of killing RV channel signals
    - Bias init −1.0: sigmoid(−1.0)≈0.27, initial output≈1.27×skip
      near-identity at start, model learns to attend gradually
    - 3×3 kernel: sufficient spatial context at 28×28 and 56×56

Applied at:
    Dec4 skip (256ch, 28×28)  — deep semantic, RV shape context
    Dec3 skip (128ch, 56×56)  — mid-level, boundary refinement
    Dec2, Dec1: not applied   — higher resolution, cost not justified

Parameters: ~21 per module (essentially free)

Reference: Spatial attention design inspired by CBAM spatial branch
           (Woo et al., 2018) with residual gating modification.
"""

import torch
import torch.nn as nn


class LightweightSpatialAttention(nn.Module):
    """
    Lightweight Spatial Attention for skip connection refinement.

    Parameters
    ----------
    kernel_size : int  Conv kernel size. Default 3.
    """

    def __init__(self, kernel_size: int = 3):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels  = 2,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = True,
        )
        self.bn      = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        # Initialise bias to -1.0 so sigmoid ≈ 0.27 at start
        # → output ≈ 1.27 × skip (near-identity)
        nn.init.constant_(self.conv.bias, -1.0)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out")

    def forward(self, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        skip : (B, C, H, W)  encoder skip features

        Returns
        -------
        (B, C, H, W)  spatially refined skip features
        """
        # Channel-wise pooling → spatial descriptors
        avg_pool = skip.mean(dim=1, keepdim=True)   # (B, 1, H, W)
        max_pool = skip.max(dim=1, keepdim=True)[0]  # (B, 1, H, W)

        # Concat → conv → BN → sigmoid
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # (B, 2, H, W)
        S = self.sigmoid(self.bn(self.conv(pooled)))      # (B, 1, H, W)

        # Residual gating: preserves features, never suppresses below 1×
        return skip * (1.0 + S)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    lsa = LightweightSpatialAttention(kernel_size=3)

    configs = [
        (2, 256, 28, 28, "Dec4 skip"),
        (2, 128, 56, 56, "Dec3 skip"),
    ]

    print("LSA — shape and parameter check")
    for B, C, H, W, label in configs:
        x   = torch.randn(B, C, H, W)
        out = lsa(x)
        params = sum(p.numel() for p in lsa.parameters())
        print(f"  [{label}] Input: {x.shape} → Output: {out.shape} | "
              f"Params: {params}")
        assert out.shape == x.shape

    print("  All passed.")