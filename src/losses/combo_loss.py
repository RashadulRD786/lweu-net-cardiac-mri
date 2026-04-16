"""
Combined Loss for ACDC Cardiac MRI Segmentation.

Baseline U-Net:
    L = 0.5 × Dice + 0.5 × CE

LWEU-Net Full (Row 4):
    L = 0.4 × Dice + 0.4 × CE + 0.2 × BoundaryLoss(MYO)
    with 50-epoch linear warmup on boundary term

Ablation control:
    use_boundary_loss=False  → Rows 2 and 3 (Dice + CE only, 0.5:0.5)
    use_boundary_loss=True   → Row 4 (full LWEU-Net loss)

Boundary Loss (Kervadec et al., 2019 — MIDL):
    Computes signed distance transform of ground truth MYO boundary.
    Penalises predicted MYO probabilities weighted by distance from
    true boundary. Applied to MYO class only (index 2).

Warmup schedule:
    Boundary loss weight increases linearly from 0 to boundary_weight
    over the first warmup_epochs epochs. Prevents noisy boundary
    gradients from destabilising early training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ============================================================
# Dice Loss (unchanged from baseline)
# ============================================================

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.
    Uses softmax probabilities, macro average over classes.

    Parameters
    ----------
    num_classes        : int    Default 4.
    smooth             : float  Laplace smoothing. Default 1e-6.
    include_background : bool   Include BG in mean. Default True.
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
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, num_classes=C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        probs_flat   = probs.view(B, C, -1)
        targets_flat = targets_one_hot.view(B, C, -1)

        intersection   = (probs_flat * targets_flat).sum(dim=2)
        union          = probs_flat.sum(dim=2) + targets_flat.sum(dim=2)
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_per_class = dice_per_class.mean(dim=0)

        if not self.include_background:
            dice_per_class = dice_per_class[1:]

        return 1.0 - dice_per_class.mean()


# ============================================================
# Boundary Loss — MYO only (Kervadec et al., 2019)
# ============================================================

class BoundaryLoss(nn.Module):
    """
    Distance-weighted boundary loss for MYO class.

    L_boundary = mean( φ_GT(q) × s_θ(q) )

    φ_GT : signed distance transform of GT MYO boundary
           positive outside, negative inside
    s_θ  : predicted MYO softmax probability

    Parameters
    ----------
    myo_class_idx : int  MYO class index. Default 2.
    """

    def __init__(self, myo_class_idx: int = 2):
        super().__init__()
        self.myo_class_idx    = myo_class_idx
        self._scipy_available = None

    def _check_scipy(self) -> bool:
        if self._scipy_available is None:
            try:
                from scipy.ndimage import distance_transform_edt
                self._scipy_available = True
            except ImportError:
                logger.warning(
                    "scipy not installed — BoundaryLoss returns 0. "
                    "Install with: pip install scipy"
                )
                self._scipy_available = False
        return self._scipy_available

    def _compute_distance_map(self, mask: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt
        mask = mask.astype(bool)

        if not mask.any():
            return np.zeros(mask.shape, dtype=np.float32)

        dist_outside = distance_transform_edt(~mask)
        dist_inside  = distance_transform_edt(mask)
        return (dist_outside - dist_inside).astype(np.float32)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if not self._check_scipy():
            return torch.tensor(0.0, device=logits.device, requires_grad=False)

        # MYO predicted probability: (B, H, W)
        myo_prob   = F.softmax(logits, dim=1)[:, self.myo_class_idx, :, :]
        targets_np = targets.detach().cpu().numpy()

        dist_maps = []
        for b in range(targets.shape[0]):
            myo_mask = (targets_np[b] == self.myo_class_idx)
            dist_maps.append(self._compute_distance_map(myo_mask))

        dist_tensor = torch.from_numpy(
            np.stack(dist_maps, axis=0)
        ).to(logits.device)

        return (dist_tensor * myo_prob).mean()


# ============================================================
# Combined Loss
# ============================================================

class CombinedLoss(nn.Module):
    """
    Combined Dice + CE loss with optional boundary term for LWEU-Net.

    Baseline mode (use_boundary_loss=False):
        L = 0.5 × Dice + 0.5 × CE

    LWEU-Net Full (use_boundary_loss=True):
        L = dice_weight × Dice + ce_weight × CE
            + warmup(epoch) × boundary_weight × BoundaryLoss(MYO)

    Ablation mapping:
        Row 2 — LWEU-Net Base  : use_boundary_loss=False
        Row 3 — + Decoder ECA  : use_boundary_loss=False
        Row 4 — LWEU-Net Full  : use_boundary_loss=True

    IMPORTANT: Call loss.set_epoch(epoch) at the start of each
    training epoch in trainer.py to update the warmup weight.

    Parameters
    ----------
    num_classes         : int     Default 4.
    dice_weight         : float   Default 0.5 (baseline) or 0.4 (LWEU-Net).
    ce_weight           : float   Default 0.5 (baseline) or 0.4 (LWEU-Net).
    smooth              : float   Default 1e-6.
    class_weights       : tensor  Optional per-class CE weights.
    use_boundary_loss   : bool    Default False.
    boundary_weight     : float   Max boundary weight. Default 0.2.
    warmup_epochs       : int     Warmup duration. Default 50.
    myo_class_idx       : int     Default 2.
    """

    def __init__(
        self,
        num_classes:       int   = 4,
        dice_weight:       float = 0.5,
        ce_weight:         float = 0.5,
        smooth:            float = 1e-6,
        class_weights:     torch.Tensor = None,
        use_boundary_loss: bool  = False,
        boundary_weight:   float = 0.2,
        warmup_epochs:     int   = 50,
        myo_class_idx:     int   = 2,
    ):
        super().__init__()

        self.dice_weight            = dice_weight
        self.ce_weight              = ce_weight
        self.use_boundary_loss      = use_boundary_loss
        self.boundary_weight        = boundary_weight
        self.warmup_epochs          = warmup_epochs
        self._current_boundary_weight = 0.0
        self._current_epoch           = 0

        self.dice_loss = DiceLoss(
            num_classes        = num_classes,
            smooth             = smooth,
            include_background = True,
        )
        self.ce_loss = nn.CrossEntropyLoss(
            weight    = class_weights,
            reduction = "mean",
        )

        if use_boundary_loss:
            self.boundary_loss_fn = BoundaryLoss(myo_class_idx=myo_class_idx)
            logger.info(
                f"BoundaryLoss enabled — MYO class {myo_class_idx}, "
                f"max weight={boundary_weight}, warmup={warmup_epochs} epochs"
            )

    def set_epoch(self, epoch: int):
        """
        Update boundary loss warmup weight.
        Must be called at the start of each training epoch.

        Parameters
        ----------
        epoch : int  Current epoch (1-indexed)
        """
        self._current_epoch = epoch
        if self.use_boundary_loss:
            progress = min(epoch / self.warmup_epochs, 1.0)
            self._current_boundary_weight = self.boundary_weight * progress

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple:
        """
        Parameters
        ----------
        logits  : (B, C, H, W)
        targets : (B, H, W)

        Returns
        -------
        total_loss : torch.Tensor
        loss_dict  : dict  {total, dice, ce, boundary}
        """
        dice  = self.dice_loss(logits, targets)
        ce    = self.ce_loss(logits, targets)
        total = self.dice_weight * dice + self.ce_weight * ce

        loss_dict = {
            "total":    total.item(),
            "dice":     dice.item(),
            "ce":       ce.item(),
            "boundary": 0.0,
        }

        if self.use_boundary_loss and self._current_boundary_weight > 0:
            boundary = self.boundary_loss_fn(logits, targets)
            weighted = self._current_boundary_weight * boundary
            total    = total + weighted
            loss_dict["boundary"] = boundary.item()
            loss_dict["total"]    = total.item()

        return total, loss_dict