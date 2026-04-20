"""
LWEU-Net V2 Encoder — Enhanced LRHE
 
Architecture:
    Level 1 : StandardResidualBlock  (1   →  32,  224×224)
    Level 2 : StandardResidualBlock  (32  →  64,  112×112)
    Level 3 : EnhancedBlock (r=3)    (64  → 128,   56×56)
    Level 4 : EnhancedBlock (r=4,    (128 → 256,   28×28)
               + context gate)
 
Changes from V1 (original LRHE):
    - Levels 3-4 now use EnhancedBlock instead of DepthwiseSepResidualBlock
    - EnhancedBlock adds non-linear channel MLP and optional global context
    - Enc4 has global context gate for RV shape understanding
    - Enc3 has no context gate (not needed at 56×56 resolution)
"""
 
import torch
import torch.nn as nn
 
from src.models.lweunet.encoder import StandardResidualBlock
from src.models.lweunet.enhanced_block import EnhancedBlock
 
 
class EncoderLevelV2(nn.Module):
    """
    Single encoder level: block → skip + MaxPool.
 
    Parameters
    ----------
    in_channels  : int
    out_channels : int
    block        : nn.Module  Pre-constructed block instance
    """
 
    def __init__(self, in_channels: int, out_channels: int, block: nn.Module):
        super().__init__()
        self.block = block
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
 
    def forward(self, x: torch.Tensor):
        skip = self.block(x)
        down = self.pool(skip)
        return skip, down
 
 
class LRHEEncoderV2(nn.Module):
    """
    Enhanced Lightweight Residual Hybrid Encoder (LRHE V2).
 
    Levels 1-2 : StandardResidualBlock (unchanged — fine boundary detection)
    Levels 3-4 : EnhancedBlock (non-linear channel mixing + global context at L4)
 
    Parameters
    ----------
    in_channels  : int   Input channels. Default 1.
    base_filters : int   Base filter count. Default 32.
 
    Returns
    -------
    skips : [skip1, skip2, skip3, skip4]
    out   : encoder output fed to bottleneck (B, 256, 14, 14)
    """
 
    def __init__(self, in_channels: int = 1, base_filters: int = 32):
        super().__init__()
 
        f = base_filters   # 32
 
        self.level1 = EncoderLevelV2(
            in_channels, f,
            StandardResidualBlock(in_channels, f),
        )
        self.level2 = EncoderLevelV2(
            f, f * 2,
            StandardResidualBlock(f, f * 2),
        )
        self.level3 = EncoderLevelV2(
            f * 2, f * 4,
            EnhancedBlock(f * 2, f * 4, expansion=3, dilation=1, use_context=False),
        )
        self.level4 = EncoderLevelV2(
            f * 4, f * 8,
            EnhancedBlock(f * 4, f * 8, expansion=4, dilation=1, use_context=True),
        )
 
    def forward(self, x: torch.Tensor):
        skip1, x = self.level1(x)   # (B, 32,  224, 224)
        skip2, x = self.level2(x)   # (B, 64,  112, 112)
        skip3, x = self.level3(x)   # (B, 128,  56,  56)
        skip4, x = self.level4(x)   # (B, 256,  28,  28)
        return [skip1, skip2, skip3, skip4], x
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
 
# ============================================================
# Sanity check
# ============================================================
 
if __name__ == "__main__":
    encoder = LRHEEncoderV2(in_channels=1, base_filters=32)
    x = torch.randn(2, 1, 224, 224)
    skips, out = encoder(x)
 
    print("LRHEEncoderV2 — shape check")
    for i, s in enumerate(skips):
        print(f"  Skip {i+1} : {s.shape}")
    print(f"  Output : {out.shape}")
    print(f"  Params : {encoder.count_parameters():,}")
 