"""
LWEU-Net V2 Bottleneck — Enhanced Dilated Block

Uses EnhancedBlock with dilation=2 and global context gate.

Changes from V1:
    - Replaces DepthwiseSepResidualBlock with EnhancedBlock
    - Keeps dilation=2 for expanded receptive field
    - Adds global context gate for apical/basal slice understanding
    - expansion=4 for maximum feature richness at bottleneck
"""

import torch
import torch.nn as nn

from src.models.lweunet.enhanced_block import EnhancedBlock


class DilatedBottleneckV2(nn.Module):
    """
    Enhanced Dilated Bottleneck.

    Parameters
    ----------
    in_channels  : int    Default 256
    out_channels : int    Default 512
    dilation     : int    Default 2
    expansion    : int    Default 4
    dropout_p    : float  Default 0.5
    """

    def __init__(
        self,
        in_channels:  int   = 256,
        out_channels: int   = 512,
        dilation:     int   = 2,
        expansion:    int   = 4,
        dropout_p:    float = 0.5,
    ):
        super().__init__()

        self.block = EnhancedBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            expansion    = expansion,
            dilation     = dilation,
            use_context  = True,
        )
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.block(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    bn = DilatedBottleneckV2(256, 512, dilation=2, expansion=4)
    x  = torch.randn(2, 256, 14, 14)
    y  = bn(x)
    print(f"Bottleneck V2 | Input: {x.shape} → Output: {y.shape}")
    print(f"Params: {bn.count_parameters():,}")
    assert y.shape == (2, 512, 14, 14)
    print("Passed.")