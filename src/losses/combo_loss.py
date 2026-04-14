"""
Combined Dice + Cross-Entropy loss for multi-class cardiac MRI segmentation.

Loss = 0.5 × DiceLoss + 0.5 × CrossEntropyLoss

Justified by:
- Yang 2020 (LFCN)  : CE + Dice combined, equal weight
- Singh 2023        : Dice + CE combined
- Wijesinghe 2025   : 0.6 CE + 0.4 Dice
- UwU-Net 2025      : 0.5 BCE + 0.5 Dice
- Cui 2021 (AID)    : "background pixels vastly outnumber cardiac structure pixels;
                       CE handles per-pixel accuracy, Dice handles class imbalance"

Design notes:
  - DiceLoss operates on softmax probabilities (NOT raw logits)
  - CrossEntropyLoss operates on raw logits (applies log-softmax internally)
  - Background class (index 0) is INCLUDED in both losses — consistent with
    all Tier 1 comparison papers that report 3-class foreground Dice only at
    evaluation time, but train with all 4 classes
  - Smooth = 1e-6 in Dice to prevent division by zero on empty classes
    (rare but possible for RV in apical/basal slices)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.

    Computes per-class Dice and averages across classes (macro average).
    Uses softmax probabilities, not hard predictions.

    Parameters
    ----------
    num_classes : int
        Number of output classes including background. Default 4.
    smooth : float
        Laplace smoothing constant to prevent 0/0. Default 1e-6.
    include_background : bool
        If True, background class is included in the mean. Default True
        (consistent with training on all 4 classes).
    """

    def __init__(
        self,
        num_classes: int = 4,
        smooth: float = 1e-6,
        include_background: bool = True,
    ):
        super().__init__()
        self.num_classes        = num_classes
        self.smooth             = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)  raw network output (before softmax)
        targets : (B, H, W)     integer class labels, values in [0, C-1]

        Returns
        -------
        torch.Tensor : scalar mean Dice loss (lower = better)
        """
        B, C, H, W = logits.shape

        # Softmax probabilities: (B, C, H, W)
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=C)   # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        # Flatten spatial dims: (B, C, H*W)
        probs_flat   = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)

        # Per-class Dice: sum over pixels, then mean over batch
        intersection = (probs_flat * targets_flat).sum(dim=2)   # (B, C)
        union        = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)  # (B, C)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # Mean over batch
        dice_per_class = dice_per_class.mean(dim=0)  # (C,)

        # Optionally exclude background
        if not self.include_background:
            dice_per_class = dice_per_class[1:]

        # Loss = 1 - mean Dice
        return 1.0 - dice_per_class.mean()


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss.

    L_total = dice_weight × DiceLoss + ce_weight × CrossEntropyLoss

    Default weights: 0.5 / 0.5 (consensus choice from literature review).

    Parameters
    ----------
    num_classes : int
        Number of output classes. Default 4.
    dice_weight : float
        Weight for Dice loss component. Default 0.5.
    ce_weight : float
        Weight for CrossEntropy loss component. Default 0.5.
    smooth : float
        Smoothing for Dice numerator/denominator. Default 1e-6.
    class_weights : torch.Tensor or None
        Optional per-class weights for CrossEntropyLoss.
        Useful if background:foreground imbalance causes instability.
        Shape: (num_classes,). Default None (uniform weighting).
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 0.5,
        ce_weight: float = 0.5,
        smooth: float = 1e-6,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()

        assert abs(dice_weight + ce_weight - 1.0) < 1e-6, \
            f"dice_weight + ce_weight must equal 1.0, got {dice_weight + ce_weight}"

        self.dice_weight = dice_weight
        self.ce_weight   = ce_weight

        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            include_background=True,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights,   # None → uniform; pass tensor for weighted CE
            reduction="mean",
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        logits  : (B, C, H, W) raw network output
        targets : (B, H, W)    integer labels in [0, C-1]

        Returns
        -------
        total_loss : torch.Tensor  scalar, for .backward()
        loss_dict  : dict          {'total': float, 'dice': float, 'ce': float}
                                   Useful for logging individual components.
        """
        dice = self.dice_loss(logits, targets)
        ce   = self.ce_loss(logits, targets)

        total = self.dice_weight * dice + self.ce_weight * ce

        loss_dict = {
            "total": total.item(),
            "dice":  dice.item(),
            "ce":    ce.item(),
        }

        return total, loss_dict