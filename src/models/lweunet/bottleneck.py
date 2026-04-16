"""
LWEU-Net Bottleneck — Dilated Depthwise Separable Residual Block

Architecture:
    Input  : (B, 256, 14, 14)  — from encoder Level 4 pooling
    Output : (B, 512, 14, 14)  — fed to decoder Level 4

Design:
    - Uses DepthwiseSepResidualBlock with dilation=2
    - Dilation expands receptive field from 3×3 to 5×5 effective area
      without increasing parameters or reducing spatial resolution
    - Captures long-range dependencies and global contextual information
      critical for handling anatomical variability across apical/basal slices
    - Dropout p=0.5 applied after the residual block for regularisation
      consistent with baseline U-Net bottleneck design

Justification:
    Standard bottleneck (baseline U-Net) uses a fixed 3×3 receptive field
    at 14×14 resolution — too local to understand full cardiac context.
    Dilation=2 allows the bottleneck to see a wider area, helping the model
    distinguish apical slices (small/absent structures) from mid-ventricular
    slices (large clear structures) without adding parameters.
"""

import torch
import torch.nn as nn

from src.models.lweunet.encoder import DepthwiseSepResidualBlock


class DilatedBottleneck(nn.Module):
    """
    Dilated Depthwise Separable Residual Bottleneck.

    Parameters
    ----------
    in_channels  : int    Channels from encoder output. Default 256.
    out_channels : int    Bottleneck output channels. Default 512.
    dilation     : int    Dilation rate for depthwise conv. Default 2.
    dropout_p    : float  Dropout probability. Default 0.5.
    """

    def __init__(
        self,
        in_channels: int  = 256,
        out_channels: int = 512,
        dilation: int     = 2,
        dropout_p: float  = 0.5,
    ):
        super().__init__()

        self.block = DepthwiseSepResidualBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            dilation     = dilation,
        )

        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (B, 256, 14, 14)  encoder output

        Returns
        -------
        out : (B, 512, 14, 14)  fed to decoder
        """
        out = self.block(x)
        out = self.dropout(out)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Sanity check
# ============================================================

if __name__ == "__main__":
    bottleneck = DilatedBottleneck(
        in_channels  = 256,
        out_channels = 512,
        dilation     = 2,
        dropout_p    = 0.5,
    )

    x   = torch.randn(2, 256, 14, 14)
    out = bottleneck(x)

    print("Dilated Bottleneck — shape check")
    print(f"  Input      : {x.shape}")
    print(f"  Output     : {out.shape}")
    print(f"  Parameters : {bottleneck.count_parameters():,}")

    assert out.shape == (2, 512, 14, 14), "Output shape mismatch!"
    print("  Sanity check passed.")