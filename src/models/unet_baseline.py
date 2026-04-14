"""
Vanilla U-Net Baseline for ACDC Cardiac MRI Segmentation.

Architecture (Ronneberger et al., 2015) with modern additions:
  - BatchNorm after every Conv (absent in 2015 original; added per Tesfaye 2023,
    Yang 2020, Singh 2023, Wijesinghe 2025 — universal in modern ACDC implementations)
  - ConvTranspose2d upsampling (faithful to original; Tesfaye 2023)
  - Dropout p=0.5 at bottleneck only (Ronneberger 2015, ARW-Net)
  - He normal weight initialization (Yang 2020, Ronneberger 2015)
  - Softmax output (multi-class; consistent with CrossEntropyLoss usage)

Filter progression : 64 → 128 → 256 → 512 → 1024 (bottleneck)
Encoder levels     : 4
Input              : (B, 1, 224, 224)  — 1 channel grayscale MRI
Output             : (B, 4, 224, 224)  — 4-class logits (BG, RV, MYO, LV)
Parameters         : ~31M (standard Ronneberger 2015 parameter count)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Two consecutive Conv3×3 → BatchNorm → ReLU blocks.

    This is the fundamental repeating unit of U-Net, used at every
    encoder level, bottleneck, and decoder level.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder block: ConvBlock → output feature map (for skip) + MaxPool (for next level).

    Returns both the skip connection feature map and the downsampled output.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv    = ConvBlock(in_channels, out_channels)
        self.pool    = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)      # saved for skip connection
        down = self.pool(skip)   # passed to next encoder level
        return skip, down


class DecoderBlock(nn.Module):
    """
    Decoder block: ConvTranspose2d upsample → concatenate skip → ConvBlock.

    Upsampling via ConvTranspose2d (kernel=2, stride=2) follows the
    original U-Net (Ronneberger 2015) and Tesfaye 2023.

    Parameters
    ----------
    in_channels   : channels coming from lower decoder level (before upsample)
    skip_channels : channels from the corresponding encoder skip connection
    out_channels  : output channels after ConvBlock
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(
            in_channels, in_channels // 2,
            kernel_size=2, stride=2
        )
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)   # concatenate along channel axis
        return self.conv(x)


# ---------------------------------------------------------------------------
# Full U-Net
# ---------------------------------------------------------------------------

class UNetBaseline(nn.Module):
    """
    Vanilla U-Net Baseline.

    Parameters
    ----------
    in_channels  : int  Input channels. Default 1 (grayscale MRI).
    num_classes  : int  Output classes. Default 4 (BG, RV, MYO, LV).
    base_filters : int  Filters at first encoder level. Default 64.
    dropout_p    : float  Dropout probability at bottleneck. Default 0.5.

    Architecture summary
    --------------------
    Encoder:
        E1: ConvBlock(1  →  64)  → skip1 + MaxPool
        E2: ConvBlock(64 → 128)  → skip2 + MaxPool
        E3: ConvBlock(128→ 256)  → skip3 + MaxPool
        E4: ConvBlock(256→ 512)  → skip4 + MaxPool
    Bottleneck:
        ConvBlock(512 → 1024) → Dropout(0.5)
    Decoder:
        D4: ConvTranspose(1024→512) + cat(skip4) → ConvBlock(1024→512)
        D3: ConvTranspose(512→256)  + cat(skip3) → ConvBlock(512→256)
        D2: ConvTranspose(256→128)  + cat(skip2) → ConvBlock(256→128)
        D1: ConvTranspose(128→64)   + cat(skip1) → ConvBlock(128→64)
    Output:
        Conv1×1(64 → num_classes) → raw logits
    """

    def __init__(
        self,
        in_channels: int  = 1,
        num_classes: int  = 4,
        base_filters: int = 64,
        dropout_p: float  = 0.5,
    ):
        super().__init__()

        f = base_filters   # shorthand: f=64, 2f=128, 4f=256, 8f=512, 16f=1024

        # --- Encoder ---
        self.enc1 = EncoderBlock(in_channels, f)       #  1 →  64
        self.enc2 = EncoderBlock(f,           f * 2)   # 64 → 128
        self.enc3 = EncoderBlock(f * 2,       f * 4)   # 128→ 256
        self.enc4 = EncoderBlock(f * 4,       f * 8)   # 256→ 512

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ConvBlock(f * 8, f * 16),                  # 512 → 1024
            nn.Dropout2d(p=dropout_p),
        )

        # --- Decoder ---
        # DecoderBlock(in_channels, skip_channels, out_channels)
        self.dec4 = DecoderBlock(f * 16, f * 8,  f * 8)   # 1024+512 → 512
        self.dec3 = DecoderBlock(f * 8,  f * 4,  f * 4)   # 512+256  → 256
        self.dec2 = DecoderBlock(f * 4,  f * 2,  f * 2)   # 256+128  → 128
        self.dec1 = DecoderBlock(f * 2,  f,       f)       # 128+64   →  64

        # --- Output ---
        self.output_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        # --- Weight initialisation (He normal for ReLU) ---
        self._init_weights()

    def _init_weights(self):
        """He normal initialisation for Conv2d; constant for BatchNorm."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W)  Normalised grayscale MRI slice

        Returns
        -------
        logits : (B, num_classes, H, W)  Raw logits (apply softmax for probabilities)
        """
        # Encoder
        skip1, x = self.enc1(x)   # skip1: (B, 64,  H,    W   )
        skip2, x = self.enc2(x)   # skip2: (B, 128, H/2,  W/2 )
        skip3, x = self.enc3(x)   # skip3: (B, 256, H/4,  W/4 )
        skip4, x = self.enc4(x)   # skip4: (B, 512, H/8,  W/8 )

        # Bottleneck
        x = self.bottleneck(x)    # (B, 1024, H/16, W/16)

        # Decoder
        x = self.dec4(x, skip4)   # (B, 512,  H/8,  W/8 )
        x = self.dec3(x, skip3)   # (B, 256,  H/4,  W/4 )
        x = self.dec2(x, skip2)   # (B, 128,  H/2,  W/2 )
        x = self.dec1(x, skip1)   # (B, 64,   H,    W   )

        return self.output_conv(x)  # (B, num_classes, H, W)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = UNetBaseline(in_channels=1, num_classes=4, base_filters=64)
    x     = torch.randn(2, 1, 224, 224)
    out   = model(x)

    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    assert out.shape == (2, 4, 224, 224), "Output shape mismatch!"
    print("Sanity check passed.")