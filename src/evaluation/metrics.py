"""
Evaluation metrics for ACDC cardiac MRI segmentation.

Segmentation metrics (per class + mean foreground):
    - Dice Score (DSC)
    - IoU / Jaccard Index
    - Precision
    - Recall
    - HD95 (95th percentile Hausdorff Distance in mm)

Efficiency metrics (model-level, computed once):
    - Number of parameters
    - FLOPs  (requires: pip install thop)
    - Inference time (ms per slice)

Class mapping: 0=BG, 1=RV, 2=MYO, 3=LV
Foreground mean excludes BG — consistent with Yang 2020, Singh 2023, Tesfaye 2023.

HD95 reference (Gomathi 2022 — only Tier 1 source with U-Net HD on ACDC):
    LV  : ED=6.7mm,  ES=9.5mm
    RV  : ED=10.5mm, ES=14.8mm
    MYO : ED=8.4mm,  ES=10.2mm
"""

import time
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ============================================================
# HD95 helper
# ============================================================

def _compute_hd95_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute 95th percentile Hausdorff Distance between two binary masks.

    Parameters
    ----------
    pred   : (H, W) bool numpy array — predicted mask for one class
    target : (H, W) bool numpy array — ground truth mask for one class

    Returns
    -------
    float : HD95 in pixels.
            0.0  if both masks are empty (structure absent in both).
            diagonal of image if one is empty and other is not (worst case).
    """
    try:
        from scipy.ndimage import distance_transform_edt, binary_erosion
    except ImportError:
        logger.warning("scipy not installed — HD95 unavailable. "
                       "Install with: pip install scipy")
        return float("nan")

    pred   = pred.astype(bool)
    target = target.astype(bool)

    # Both empty — no structure present, no error
    if not pred.any() and not target.any():
        return 0.0

    # One empty — worst-case penalty (diagonal of 224×224 image)
    if not pred.any() or not target.any():
        #return float(np.sqrt(pred.shape[0]**2 + pred.shape[1]**2))
         return float("nan")

    # Surface = mask XOR eroded mask
    struct         = np.ones((3, 3), dtype=bool)
    pred_surface   = pred   ^ binary_erosion(pred,   structure=struct)
    target_surface = target ^ binary_erosion(target, structure=struct)

    # Distance transform from each surface
    dist_pred_to_target = distance_transform_edt(~target_surface)
    dist_target_to_pred = distance_transform_edt(~pred_surface)

    # HD95: 95th percentile of all directed distances
    d1 = dist_pred_to_target[pred_surface]
    d2 = dist_target_to_pred[target_surface]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


# ============================================================
# Per-batch segmentation metrics (used inside training loop)
# ============================================================

def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    smooth: float = 1e-6,
) -> dict:
    """
    Compute Dice, IoU, Precision, Recall from hard predictions.

    HD95 is NOT computed here — too slow per batch.
    It is computed in evaluate_on_test_set over the full test set.

    Parameters
    ----------
    logits  : (B, C, H, W)  raw network output
    targets : (B, H, W)     integer class labels in [0, C-1]
    smooth  : float         smoothing constant to avoid 0/0

    Returns
    -------
    dict with per-class and mean foreground metrics:
        dice_{bg,rv,myo,lv}, mean_dice
        iou_{bg,rv,myo,lv},  mean_iou
        prec_{bg,rv,myo,lv}, mean_precision
        rec_{bg,rv,myo,lv},  mean_recall
    """
    class_names = {0: "bg", 1: "rv", 2: "myo", 3: "lv"}

    preds   = logits.argmax(dim=1).detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    result = {}
    dice_fg, iou_fg, prec_fg, rec_fg = [], [], [], []

    for c in range(num_classes):
        name     = class_names[c]
        pred_c   = (preds   == c).astype(np.float32)
        target_c = (targets == c).astype(np.float32)

        tp = (pred_c * target_c).sum()
        fp = (pred_c * (1 - target_c)).sum()
        fn = ((1 - pred_c) * target_c).sum()

        dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
        iou  = (tp + smooth)       / (tp + fp + fn + smooth)
        prec = (tp + smooth)       / (tp + fp + smooth)
        rec  = (tp + smooth)       / (tp + fn + smooth)

        result[f"dice_{name}"] = float(dice)
        result[f"iou_{name}"]  = float(iou)
        result[f"prec_{name}"] = float(prec)
        result[f"rec_{name}"]  = float(rec)

        if c > 0:
            dice_fg.append(dice)
            iou_fg.append(iou)
            prec_fg.append(prec)
            rec_fg.append(rec)

    result["mean_dice"]      = float(np.mean(dice_fg))
    result["mean_iou"]       = float(np.mean(iou_fg))
    result["mean_precision"] = float(np.mean(prec_fg))
    result["mean_recall"]    = float(np.mean(rec_fg))

    return result


# ============================================================
# Full test-set evaluation (run once on best checkpoint)
# ============================================================

@torch.no_grad()
def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_classes: int = 4,
    pixel_spacing_mm: float = 1.5,
) -> dict:
    """
    Run full evaluation on the test set after training completes.

    Computes Dice, IoU, Precision, Recall, and HD95 over all test slices.
    HD95 is computed per slice then averaged — standard approach in
    medical image segmentation literature.

    Parameters
    ----------
    model            : trained model with best checkpoint loaded
    test_loader      : DataLoader for test split (augment=False)
    device           : torch.device
    pixel_spacing_mm : resampling target spacing (default=1.5mm).
                       HD95 in pixels × spacing = HD95 in mm.

    Returns
    -------
    dict with all per-class metrics + hd95 + inference_time_ms
    """
    model.eval()

    class_names = {0: "bg", 1: "rv", 2: "myo", 3: "lv"}

    metric_keys = (
        [f"dice_{n}" for n in class_names.values()] +
        [f"iou_{n}"  for n in class_names.values()] +
        [f"prec_{n}" for n in class_names.values()] +
        [f"rec_{n}"  for n in class_names.values()] +
        ["mean_dice", "mean_iou", "mean_precision", "mean_recall"]
    )
    accum     = {k: 0.0 for k in metric_keys}
    n_batches = 0

    # HD95 collected per slice, averaged at the end
    hd95_slices = {"rv": [], "myo": [], "lv": []}

    total_time   = 0.0
    total_slices = 0

    for images, masks in test_loader:
        images = images.to(device)
        masks  = masks.to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_time   += (t1 - t0)
        total_slices += images.shape[0]

        # Overlap metrics
        batch_metrics = compute_segmentation_metrics(logits, masks, num_classes)
        for k in metric_keys:
            accum[k] += batch_metrics[k]
        n_batches += 1

        # HD95 per slice per foreground class
        preds_np   = logits.argmax(dim=1).detach().cpu().numpy()
        targets_np = masks.detach().cpu().numpy()

        for b in range(preds_np.shape[0]):
            for c, name in [(1, "rv"), (2, "myo"), (3, "lv")]:
                hd_px = _compute_hd95_binary(
                    preds_np[b]   == c,
                    targets_np[b] == c,
                )
                hd95_slices[name].append(hd_px * pixel_spacing_mm)

    # Average overlap metrics over batches
    results = {k: v / n_batches for k, v in accum.items()}

    # Average HD95 over slices (exclude nan from missing scipy)
    for name in ["rv", "myo", "lv"]:
        vals = [v for v in hd95_slices[name] if not np.isnan(v)]
        results[f"hd95_{name}"] = float(np.mean(vals)) if vals else float("nan")

    results["mean_hd95"] = float(np.mean([
        results["hd95_rv"],
        results["hd95_myo"],
        results["hd95_lv"],
    ]))

    results["inference_time_ms"] = (total_time / total_slices) * 1000.0

    return results


# ============================================================
# Model efficiency metrics (computed once)
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(
    model: nn.Module,
    input_size: tuple = (1, 1, 224, 224),
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    """
    Compute GFLOPs for a single forward pass using thop.
    Install with:  pip install thop
    Returns GFLOPs (float) or None if thop is not installed.
    """
    try:
        from thop import profile
        dummy = torch.randn(*input_size).to(device)
        model = model.to(device)
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
        logger.info(f"FLOPs: {gflops:.2f} GFLOPs")
        return gflops
    except ImportError:
        logger.warning("thop not installed. Run: pip install thop")
        return None


def get_efficiency_summary(
    model: nn.Module,
    input_size: tuple = (1, 1, 224, 224),
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Returns parameter count and FLOPs in a single dict."""
    params = count_parameters(model)
    gflops = compute_flops(model, input_size, device)
    summary = {
        "num_parameters":   params,
        "num_parameters_M": round(params / 1e6, 2),
        "gflops":           gflops,
    }
    logger.info(f"Parameters: {params:,} ({summary['num_parameters_M']}M)")
    return summary


# ============================================================
# Pretty-print results table
# ============================================================

def print_results_table(metrics: dict, model_name: str = "Model"):
    """Print a formatted results summary to console and return as string."""
    sep  = "=" * 62
    rows = []
    rows.append(f"\n{sep}")
    rows.append(f"  Results — {model_name}")
    rows.append(sep)
    rows.append(f"  {'Metric':<22} {'BG':>7} {'RV':>7} {'MYO':>7} {'LV':>7} {'Mean':>7}")
    rows.append("-" * 62)

    label_map = {
        "dice": "Dice",
        "iou":  "IoU",
        "prec": "Precision",
        "rec":  "Recall",
    }
    mean_map = {
        "dice": "mean_dice",
        "iou":  "mean_iou",
        "prec": "mean_precision",
        "rec":  "mean_recall",
    }

    for key, label in label_map.items():
        bg   = metrics.get(f"{key}_bg",  0.0)
        rv   = metrics.get(f"{key}_rv",  0.0)
        myo  = metrics.get(f"{key}_myo", 0.0)
        lv   = metrics.get(f"{key}_lv",  0.0)
        mean = metrics.get(mean_map[key], 0.0)
        rows.append(
            f"  {label:<22} {bg:>7.4f} {rv:>7.4f} {myo:>7.4f} {lv:>7.4f} {mean:>7.4f}"
        )

    # HD95 row — no BG column, values in mm
    if "hd95_rv" in metrics:
        rows.append("-" * 62)
        rv   = metrics.get("hd95_rv",  float("nan"))
        myo  = metrics.get("hd95_myo", float("nan"))
        lv   = metrics.get("hd95_lv",  float("nan"))
        mean = metrics.get("mean_hd95", float("nan"))

        def fmt(v):
            return f"{v:>7.2f}" if not np.isnan(v) else "    nan"

        rows.append(
            f"  {'HD95 (mm)':<22} {'  —':>7} {fmt(rv)} {fmt(myo)} {fmt(lv)} {fmt(mean)}"
        )

    rows.append("-" * 62)
    if "inference_time_ms" in metrics:
        rows.append(f"  Inference time  : {metrics['inference_time_ms']:.3f} ms/slice")
    if "num_parameters_M" in metrics:
        rows.append(f"  Parameters      : {metrics['num_parameters_M']}M")
    if metrics.get("gflops"):
        rows.append(f"  FLOPs           : {metrics['gflops']:.2f} GFLOPs")
    rows.append(sep + "\n")

    output = "\n".join(rows)
    print(output)
    return output